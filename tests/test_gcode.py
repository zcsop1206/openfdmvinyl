import math
import numpy as np
import pytest

from openvinyl.printer import get_profile
from openvinyl.groove import GrooveParams, build_groove
from openvinyl.dsp import test_signal as make_signal, condition
from openvinyl.gcode import emit_groove, BLOCK_START, BLOCK_END
from openvinyl.preview import parse_block, decode
from openvinyl.splice import analyze_base, splice


def _small_record(printer, signal="tone:440", seconds=2.0, **kw):
    fs = 44100
    a = condition(make_signal(signal, seconds, fs), fs, lpf=681)
    gp = GrooveParams(outer_groove_radius=60.0, inner_groove_radius=40.0,
                      record_diameter=140.0, lead_in_turns=0.3, lead_out_turns=0.3, **kw)
    return build_groove(a, fs, gp, printer)


def test_emit_and_validate():
    printer = get_profile("p1s")
    path = _small_record(printer)
    lines, rep = emit_groove(path, printer, z_top=1.0)
    assert rep.ok, rep.text()
    assert lines[0] == BLOCK_START and lines[-1] == BLOCK_END
    assert any(l.startswith("; OV rpm=") for l in lines)
    assert "M83 ; relative E" in lines
    n_extrude = sum(1 for l in lines if l.startswith("G1 X") and " E" in l)
    assert n_extrude == rep.stats["segments"]
    assert rep.stats["groove_print_time_min"] > 0


def test_bounds_failure_detected():
    printer = get_profile("p1s")
    path = _small_record(printer)
    _, rep = emit_groove(path, printer, z_top=1.0, center=(30.0, 128.0))
    assert not rep.ok
    names = {c["name"]: c["ok"] for c in rep.checks}
    assert names["xy_in_bounds"] is False


def test_round_trip_decode_tone():
    printer = get_profile("p1s")
    # 0.02 mm: inside the curvature limit for 440 Hz at r = 40-60 mm, 78 RPM
    path = _small_record(printer, amplitude=0.02)
    assert path.stats["curvature_limited_fraction"] == 0.0
    lines, _ = emit_groove(path, printer, z_top=1.0)
    dec = decode(parse_block(lines), fs=44100, playback_model=False)
    assert dec["recovered_amplitude_mm"] == pytest.approx(0.02, rel=0.03)
    # dominant frequency of the recovered audio is 440 Hz
    a = dec["audio"]
    n = len(a)
    spec = np.abs(np.fft.rfft(a * np.hanning(n)))
    f = np.fft.rfftfreq(n, 1 / 44100)
    assert f[np.argmax(spec[1:]) + 1] == pytest.approx(440, abs=3)


def test_round_trip_matches_internal_x():
    printer = get_profile("p1s")
    path = _small_record(printer, signal="sweep:300:600", amplitude=0.08)
    lines, _ = emit_groove(path, printer, z_top=1.0)
    dec = decode(parse_block(lines), fs=44100, playback_model=False)
    x_rt = np.interp(path.theta, dec["theta"], dec["x"])
    # 3-decimal coordinate rounding bounds the error at ~1 um
    assert np.abs(x_rt - path.x).max() < 0.002


def _synthetic_base(mode="M83", radius=70.0):
    lines = ["; HEADER", mode, "G90", "G28"]
    e = 0.0
    z = 0.0
    for _ in range(3):
        z += 0.2
        lines += [";LAYER_CHANGE", f";Z:{z:.2f}", f"G1 Z{z:.2f} F600"]
        lines.append(f"G1 X{128+radius:.3f} Y128 F9000")
        for i in range(1, 37):
            a = 2 * math.pi * i / 36
            if mode == "M83":
                lines.append(f"G1 X{128+radius*math.cos(a):.3f} Y{128+radius*math.sin(a):.3f} E0.1")
            else:
                e += 0.1
                lines.append(f"G1 X{128+radius*math.cos(a):.3f} Y{128+radius*math.sin(a):.3f} E{e:.3f}")
    lines += ["M400 ; wait for buffer", "; filament end gcode", "M104 S0", "M140 S0", "M84",
              "; nozzle_diameter = 0.4", "; layer_height = 0.2"]
    return lines


def test_analyze_base_relative():
    info = analyze_base(_synthetic_base())
    assert info["top_z"] == pytest.approx(0.6)
    assert info["center"][0] == pytest.approx(128.0, abs=0.01)
    assert info["center"][1] == pytest.approx(128.0, abs=0.01)
    assert info["extrusion_mode"] == "M83"
    assert info["end_marker"].startswith("M400")
    assert info["config"]["nozzle_diameter"] == "0.4"


def test_analyze_base_bambu_markers_ignore_purge_line():
    """Bambu Studio marks layers '; CHANGE_LAYER' and prints a purge line at the front edge in its start G-code.
    The centre must come from the disc's top layer only (Nov 22: a real export centred the groove 20 mm off)."""
    base = _synthetic_base()
    purge = ["M83", "G1 X18 Y1 Z0.8 F18000", "G1 X240 Y1 E20", "G1 X240 Y1.5 E0.1"]   # front-edge prime line
    i = base.index("G28") + 1
    base = base[:i] + purge + base[i:]
    base = ["; CHANGE_LAYER" if l == ";LAYER_CHANGE" else l.replace(" E0.", " E.") for l in base]
    info = analyze_base(base)
    assert info["last_extrusion_idx"] == max(i for i, l in enumerate(base) if " E." in l)
    assert info["center"][0] == pytest.approx(128.0, abs=0.01)
    assert info["center"][1] == pytest.approx(128.0, abs=0.01)
    assert max(info["last_layer_extent"]) == pytest.approx(140.0, abs=0.5)


def test_splice_marks_groove_as_a_layer():
    """Slicer previews draw moves by their layer/feature comments: the groove gets its own layer in the base's
    style (Nov 22: Bambu Studio's preview showed only the disc's top infill)."""
    base = ["; total layer number: 3"] + ["; CHANGE_LAYER" if l == ";LAYER_CHANGE" else l for l in _synthetic_base()]
    block = [BLOCK_START, "G1 Z2.2 F600", "G1 X1 Y1", "G1 Z0.8", "G1 X1 Y1 E0.1", BLOCK_END]   # travel lift first
    out, info = splice(base, block)
    assert info["layer_marker"] == "bambu" and info["layer_count"] == 3
    i = out.index(BLOCK_START)
    assert "; CHANGE_LAYER" in out[i - 9:i] and "; Z_HEIGHT: 0.8" in out[i - 9:i] and "M73 L4" in out[i - 9:i]
    assert out[0] == "; total layer number: 4"
    out_p, _ = splice(_synthetic_base(), block)                       # Prusa/Orca style
    j = out_p.index(BLOCK_START)
    assert ";LAYER_CHANGE" in out_p[j - 5:j] and ";Z:0.8" in out_p[j - 5:j]


def test_cura_base_centres_on_the_top_layer_and_marks_the_groove():
    """Cura marks layers ';LAYER:n' (from 0) and counts them in ';LAYER_COUNT:'. Without the marker the centre would
    come from every extrusion in the file, purge line included."""
    base = _synthetic_base()
    purge = ["M83", "G1 X18 Y1 Z0.8 F18000", "G1 X240 Y1 E20"]
    i = base.index("G28") + 1
    base = base[:i] + purge + base[i:]
    k = iter(range(3))
    base = [f";LAYER:{next(k)}" if l == ";LAYER_CHANGE" else l for l in base if not l.startswith(";Z:")]
    base = [";LAYER_COUNT:3"] + base
    info = analyze_base(base)
    assert info["layer_marker"] == "cura" and info["layer_count"] == 3
    assert info["center"][0] == pytest.approx(128.0, abs=0.01)
    assert info["center"][1] == pytest.approx(128.0, abs=0.01)
    block = [BLOCK_START, "G1 Z2.2 F600", "G1 X1 Y1", "G1 Z0.8", "G1 X1 Y1 E0.1", BLOCK_END]
    out, _ = splice(base, block)
    j = out.index(BLOCK_START)
    assert ";LAYER:3" in out[j - 4:j]
    assert out[0] == ";LAYER_COUNT:4"


def test_profile_from_json(tmp_path):
    p = tmp_path / "mine.json"
    p.write_text('{"_comment": "notes are ignored", "name": "Mine", "bed_x": 220, "bed_y": 220, '
                 '"groove_accel_max": 1500, "max_segment_rate": 200}')
    prof = get_profile(str(p))
    assert (prof.name, prof.bed_x, prof.groove_accel_max, prof.max_segment_rate) == ("Mine", 220, 1500, 200)
    assert prof.nozzle_diameter == 0.4                                  # left out: the default
    assert get_profile(str(p), nozzle=0.6).nozzle_diameter == 0.6
    (tmp_path / "typo.json").write_text('{"bed_xx": 220}')
    with pytest.raises(KeyError):
        get_profile(str(tmp_path / "typo.json"))


def test_template_profile_loads():
    import pathlib
    t = pathlib.Path(__file__).resolve().parents[1] / "printers" / "template.json"
    prof = get_profile(str(t))
    assert prof.bed_x > 0 and prof.extrusion_width > 0


def test_splice_counts_the_groove_in_time_and_progress(tmp_path):
    """The printer's clock: header totals, every M73 progress line, progress through the groove, and the .3mf's
    prediction must include the groove (Nov 22: the screen said 38 min for a 50 min print)."""
    import zipfile
    from openvinyl.splice import pack_3mf
    base = ["; HEADER_BLOCK_START", "; model printing time: 10m 0s; total estimated time: 12m 0s",
            "; total filament weight [g] : 30.00", "; filament_density: 1.24", "; filament_diameter: 1.75",
            "; HEADER_BLOCK_END", "M73 P0 R12", "M73 P50 R6"] + _synthetic_base() + ["M73 P100 R0"]
    # a 6 minute groove: 10800 mm at 30 mm/s, at one height
    block = [BLOCK_START, "G1 Z1.2 F600", "G1 X0 Y0 F1800"] + \
            [f"G1 X{100 * ((k % 2) * 2 - 1):.0f} Y0 E0.1" for k in range(1, 55)] + [BLOCK_END]
    out, info = splice(base, block)
    nl = chr(10)
    assert info["groove_time_s"] == pytest.approx((100 + 53 * 200) / 30, rel=0.01)       # 10700 mm: 5m 57s
    assert "; model printing time: 15m 57s; total estimated time: 17m 57s" in out[:3]
    assert "M73 P0 R18" in out and "M73 P33 R12" in out                 # base progress scaled to the new total
    i, j = out.index(BLOCK_START), out.index(BLOCK_END)
    inner = [l for l in out[i:j] if l.startswith("M73 P")]
    pcts = [int(l.split()[1][1:]) for l in inner]
    assert pcts[0] >= 66 and pcts[-1] == 99 and pcts == sorted(pcts)    # counts up through the groove
    assert out[-1] == "M73 P100 R0"
    assert "; total filament weight [g] : 30." in nl.join(out[:5])
    # the .3mf: prediction and weight raised by the groove's
    src = tmp_path / "base.3mf"
    with zipfile.ZipFile(src, "w") as z:
        z.writestr("Metadata/plate_1.gcode", nl.join(base))
        z.writestr("Metadata/plate_1.gcode.md5", "x")
        z.writestr("Metadata/slice_info.config", '<metadata key="prediction" value="720"/><metadata key="weight" value="30.00"/>'
                   '<filament id="1" used_m="10.00" used_g="30.00"/>')
    dst = tmp_path / "out.gcode.3mf"
    pack_3mf(str(src), nl.join(out) + nl, str(dst), info)
    with zipfile.ZipFile(dst) as z:
        cfg = z.read("Metadata/slice_info.config").decode()
        assert f'value="{720 + round(info["groove_time_s"])}"' in cfg
        assert z.read("Metadata/plate_1.gcode").decode().startswith("; HEADER_BLOCK_START")


def test_splice_absolute_mode_restored():
    base = _synthetic_base(mode="M82")
    info = analyze_base(base)
    assert info["extrusion_mode"] == "M82"
    out, _ = splice(base, [BLOCK_START, "G1 X1 Y1 E0.1", BLOCK_END], info)
    i_block = out.index(BLOCK_END)
    assert out[i_block + 1].startswith("M82")
    assert out[i_block + 2].startswith("G92 E")
    # end gcode still follows the block
    assert out.index("; filament end gcode") > i_block
    # nothing of the base was lost
    assert len(out) == len(base) + 3 + 3 + 4


def _floor_lines(lines):
    """{(layer, line): (r, theta) arrays} of the floor's extruding moves, about the bed centre (128, 128)."""
    out, key = {}, None
    for l in lines:
        if l.startswith("; OV floor line="):
            kv = dict(t.split("=") for t in l[len("; OV floor "):].split())
            key = (int(kv["layer"]), int(kv["line"]))
            out.setdefault(key, [])
        elif l.startswith("; OV bead="):
            key = None
        elif key is not None and l.startswith("G1 X") and " E" in l:
            x, y = float(l.split()[1][1:]), float(l.split()[2][1:])
            out[key].append((x - 128, y - 128))
    return {k: (np.hypot(*np.array(v).T), np.arctan2(np.array(v)[:, 1], np.array(v)[:, 0])) for k, v in out.items() if v}


def test_spiral_floor_and_partial_second_layer():
    """Floor: flat layers under the whole groove band, below the beads. Second layer: only from the chosen audio
    time on, so one disc compares one and two layers (Nov 22). The decoder still reads layer 0."""
    printer = get_profile("p1s")
    path = _small_record(printer, amplitude=0.02, groove_layers=2, floor_layers=1, extra_layers_from_s=1.0)
    lines, rep = emit_groove(path, printer, z_top=1.0)
    assert rep.ok, rep.text()
    i_floor = lines.index("; OV floor layer=0")
    i_bead = next(i for i, l in enumerate(lines) if l.startswith("; OV bead="))
    assert i_floor < i_bead
    fl = _floor_lines(lines)
    for r, _ in fl.values():
        assert np.all(np.diff(r) <= 1e-3)                              # each line spirals inward, never crossing itself
    r_all = np.concatenate([r for r, _ in fl.values()])
    assert r_all.max() > path.r_center.max() + 0.4 and r_all.min() < path.r_center.min() - 0.4
    zs = {round(float(t[1:]), 3) for l in lines if l.startswith("G1 ") for t in l.split() if t.startswith("Z")}
    assert {1.2, 1.4, 1.6} <= zs                                       # floor, bead layer 0, bead layer 1
    beads = parse_block(lines)["beads"]
    n0 = len(next(b for b in beads if b["layer"] == 0)["x"]); n1 = len(next(b for b in beads if b["layer"] == 1)["x"])
    assert 0.4 < n1 / n0 < 0.8                                          # 2 s of audio, second layer from 1 s
    dec = decode(parse_block(lines), fs=44100, playback_model=False)
    assert dec["recovered_amplitude_mm"] == pytest.approx(0.02, rel=0.05)


def test_floor_line_centred_under_every_crease():
    """Nov 24: some groove floors showed the seam between two floor lines right under the crease, because the old
    floor (one spiral at one bead width per turn) drifted against the groove. Now line 0 of the floor is the groove's
    own unmodulated spiral, so the crease sits on a line's centre all the way round and the nearest seam is half a
    line width away, under the beads."""
    printer = get_profile("p1s")
    path = _small_record(printer, amplitude=0.02, floor_layers=2)
    lines, rep = emit_groove(path, printer, z_top=1.0)
    assert rep.ok, rep.text()
    f = rep.stats["floor"]
    wf = f["line_width_mm"]
    assert f["lines_per_turn"] * wf == pytest.approx(path.geometry.pitch, abs=5e-4)
    fl = _floor_lines(lines)
    # every floor line's centre, sampled at the groove's angles, against the crease (groove centreline) radius
    th_g = np.mod(path.theta, 2 * np.pi)
    seams_off = []
    for (layer, j), (r, th) in fl.items():
        # radius of this line at each groove vertex angle, from the nearest floor vertex at that angle and turn
        order = np.argsort(np.mod(th, 2 * np.pi))
        for i in range(0, len(path.theta), 97):
            rc = path.r_spiral[i]
            near = np.abs(np.mod(th - th_g[i] + np.pi, 2 * np.pi) - np.pi) < 0.01
            if not near.any():
                continue
            d = r[near] - rc
            d = d[np.argmin(np.abs(d))]
            # distance from the crease to the nearest seam of this line (its edges are at +-wf/2)
            seams_off.append(min(abs(d - wf / 2), abs(d + wf / 2)))
    assert min(seams_off) >= wf / 2 - 0.01                              # no seam within 0.25 mm of a crease
    z_floor = sorted({round(float(t[1:]), 3) for l in lines if l.startswith("G1 Z")
                      for t in l.split() if t.startswith("Z")})
    assert 1.2 in z_floor and 1.4 in z_floor


def test_schedule_bands_change_spacing_layers_and_floor():
    """Taguchi disc: bands with their own bead spacing, layer count and floor, on one spiral of fixed pitch."""
    printer = get_profile("p1s")
    sched = [{"t1": 0.7, "spacing": 1.0, "layers": 1, "floor": 0},
             {"t1": 1.4, "spacing": 1.1, "layers": 3, "floor": 2},
             {"spacing": 1.05, "layers": 2, "floor": 1}]
    path = _small_record(printer, amplitude=0.02, schedule=sched, ramp_s=0.2)
    g = path.geometry
    assert g.bead_spacing == pytest.approx(1.1 * g.w_e)                 # the pitch is the widest band's
    ta = path.t - g.lead_in_time
    mid = lambda a, b: (ta > a) & (ta < b)
    assert np.allclose(path.spacing[mid(0.0, 0.55)], 1.0 * g.w_e)
    assert np.allclose(path.spacing[mid(0.85, 1.25)], 1.1 * g.w_e)
    assert np.allclose(path.spacing[mid(1.55, 1.9)], 1.05 * g.w_e)
    assert set(path.layers[mid(0.85, 1.25)]) == {3} and set(path.floor[mid(0.85, 1.25)]) == {2}
    ramp = mid(0.6, 0.8)                                               # spacing ramps, the floor steps a layer at a time
    assert np.all(np.diff(path.spacing[ramp]) >= -1e-12) and len(set(path.floor[ramp])) == 3
    lines, rep = emit_groove(path, printer, z_top=1.0)
    assert rep.ok, rep.text()
    # beads sit on the floor under them: layer 0 of the 2-floor band at z_top + 3 h
    zs = {round(float(t[1:]), 3) for l in lines if l.startswith("G1 X") for t in l.split() if t.startswith("Z")}
    assert {1.2, 1.6, 1.8, 2.0} <= zs | {1.2}
    # bead L's measured spacing follows the band
    (xL, yL), (xR, yR) = path.bead_xy()
    gap = np.hypot(xL - xR, yL - yR)
    assert np.median(gap[mid(0.85, 1.25)]) == pytest.approx(1.1 * g.w_e, abs=0.003)
    assert np.median(gap[mid(0.0, 0.55)]) == pytest.approx(1.0 * g.w_e, abs=0.003)
    dec = decode(parse_block(lines), fs=44100, playback_model=False)
    assert dec["recovered_amplitude_mm"] == pytest.approx(0.02, rel=0.05)
