"""
make_test_disc.py - a 7 in, 40 RPM test record for the spine turntable: tracking, speed, wow and flutter, frequency
response and distortion, with every section's amplitude inside the groove's limits where it plays.

    python make_test_disc.py        # writes ../../../out/testdisc/: the signal and its manifest, the 7 in base disc
                                    # sliced by Bambu Studio, the spliced G-code, a .gcode.3mf, and a decoded preview

Layout (Nov 22, after the first two takes). The groove sits on a spiral floor layer that follows it, so no straight
top-infill lines cross under the stylus (they played as a 0.77 mm ridge texture over the high tones). The outer half
is one bead layer, the inner half two (a groove twice as deep), to see whether wall height sets the tracking limit:
  silence            noise floor; the stylus tracking a plain groove
  333 Hz, outer      speed error from the tone's pitch, wow and flutter from its pitch wobble (once per turn = an
                     off-centre hole or platter)
  tones 698 -> 98    frequency response and distortion; high tones first, at the outer radius, where the groove is
                     fastest. Each amplitude is held to 70 % of the curvature limit at the radius it ends at
  110 Hz ladder      10, 20, 30, 40, 60 um: tracking (takes 1 and 2 stopped following between 20 and 40 um)
  -- two bead layers from here --
  198 Hz             the same tone as the outer half, to compare levels
  110 Hz ladder      the same steps, in the deeper groove
  333 Hz, inner      the reference near the label
  silence            noise floor near the label
Every tone and its first five harmonics sit at least 5 Hz from a multiple of 60 Hz, where mains hum lands, so the
analysis can tell distortion from hum (100 Hz would put its 3rd harmonic on 300 Hz hum: 98 Hz does not). Each tone starts and ends on a 10 ms raised-cosine
ramp, so a cut sine does not put a click (a slope spike) into the groove.

The analysis is analyze_test_disc.py, on the WAV from adc_capture/capture.py.
"""
import json
import math
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "out" / "testdisc"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE.parent / "print"))

DISC_D, HOLE_D, DISC_T = 177.8, 7.24, 1.0
RPM, R_OUT, R_IN = 40.0, 84.0, 46.0
A_MAX_UM = 60.0                 # the ladders' top step; the encoder's --amplitude (80 um mistracked on Nov 22, from 40)
FS = 44100
RAMP = 0.010
GAP = 0.3
KAPPA_K = 0.033                 # encoder: A f^2 <= 0.033 v^2 (0.7/s - 1/r_inner, d2x on 0.1 mm steps), mm Hz^2
MARGIN = 0.7
BED = "Textured PEI Plate"     # the P1S plate used here (Nov 22)
LADDER = (10, 20, 30, 40, 60)   # um at 110 Hz: take 1 and 2 stopped following between 20 and 40
FLOOR_LAYERS = 1                # spiral floor under the groove (the base's straight top lines played as texture)


def radius(t, pitch):
    """Groove radius at play time t (after the one lead-in turn)."""
    return R_OUT - pitch * (1.0 + t * RPM / 60.0)


def a_limit_um(f, t_end, pitch):
    v = 2 * math.pi * radius(t_end, pitch) * RPM / 60.0
    return MARGIN * KAPPA_K * v * v / (f * f) * 1000.0


def schedule(pitch):
    """[(kind, t0, t1, f, A_um)] in play time."""
    seg, t = [], 0.0

    def add(kind, dur, f=0.0, a=0.0, gap=GAP):
        nonlocal t
        if f:
            lim = a_limit_um(f, t + dur, pitch)
            assert a <= lim + 1e-9, (kind, f, a, round(lim, 1))
        seg.append((kind, round(t, 3), round(t + dur, 3), f, a))
        t += dur + (gap if f else 0.0)
    # outer half: one bead layer
    add("silence outer", 1.5)
    add("reference outer", 3.0, 333.0, 20.0, gap=0.3)
    for f, a in ((698, 4), (498, 7), (333, 15), (198, 25), (98, 25)):
        add(f"tone {f} Hz", 1.2, float(f), float(a))
    for a in LADDER:
        add(f"ladder {a} um", 1.2, 110.0, float(a))
    # inner half: two bead layers from here (a deeper groove), the 198 Hz tone and the ladder again to compare
    t_two = t
    add("tone 198 Hz (2 layers)", 1.2, 198.0, 25.0)
    for a in LADDER:
        add(f"ladder {a} um (2 layers)", 1.2, 110.0, float(a))
    add("reference inner (2 layers)", 2.0, 333.0, 8.0, gap=0.0)
    add("silence inner", 0.5)
    seg = [s_ + ((2 if s_[1] >= t_two - 1e-9 else 1),) for s_ in seg]
    return seg, t, t_two


def signal(seg, total):
    n = int(math.ceil(total * FS)) + 1
    x = np.zeros(n)
    for kind, t0, t1, f, a, _ in seg:
        if not f:
            continue
        i0, i1 = int(round(t0 * FS)), int(round(t1 * FS))
        tt = np.arange(i1 - i0) / FS
        env = np.ones_like(tt)
        k = int(RAMP * FS)
        r = 0.5 - 0.5 * np.cos(np.linspace(0, math.pi, k))
        env[:k], env[-k:] = r, r[::-1]
        x[i0:i1] = (a / A_MAX_UM) * env * np.sin(2 * math.pi * f * tt)
    return x


def base_disc():
    """Slice the 7 in blank with Bambu Studio's own presets (as print/make_plates.py does).
    Another printer: set OV_BASE_GCODE to a blank disc you sliced yourself (177.8 mm across, 1.0 mm thick, 7.24 mm
    hole, centred; docs/ADAPTING_YOUR_PRINTER.md) and it is used as is, with no .gcode.3mf."""
    import os
    if os.environ.get("OV_BASE_GCODE"):
        return Path(os.environ["OV_BASE_GCODE"]), None
    from build123d import Align, Cylinder, export_stl
    import make_plates as MP
    stl = OUT / "disc7.stl"
    d = Cylinder(DISC_D / 2, DISC_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    d -= Cylinder(HOLE_D / 2, DISC_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    export_stl(d, str(stl))
    work = OUT / "profiles"
    work.mkdir(exist_ok=True)
    prof = MP.write_profiles(work)
    # the plate on the P1S used here: the CLI has no plate option, so set it in the process the slicer loads (it picks
    # the filament's textured-plate bed temperatures from it)
    pj = json.loads(Path(prof["process"]).read_text(encoding="utf-8"))
    pj["curr_bed_type"] = BED
    Path(prof["process"]).write_text(json.dumps(pj, indent=1), encoding="utf-8")
    out3mf = OUT / "disc7_base.3mf"
    cmd = [str(MP.STUDIO / "bambu-studio.exe"), "--orient", "0", "--arrange", "1",
           "--load-settings", f"{prof['machine']};{prof['process']}", "--load-filaments", str(prof["filament"]),
           "--slice", "0", "--export-3mf", out3mf.name, "--outputdir", str(OUT), str(stl)]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    loose = OUT / "plate_1.gcode"
    if r.returncode != 0 or not loose.exists():
        print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "error" in l.lower())[-2000:])
        raise SystemExit("base disc slice failed")
    shutil.move(str(loose), OUT / "disc7_base.gcode")
    return OUT / "disc7_base.gcode", out3mf


def base_args(base, base3mf):
    """The generate flags for the base disc and the printer: --base-3mf only with Bambu's sliced project, and
    --printer from OV_PRINTER (a built-in name or a .json profile) when it is set."""
    import os
    args = ["--base", str(base)] + (["--base-3mf", str(base3mf)] if base3mf else [])
    return args + (["--printer", os.environ["OV_PRINTER"]] if os.environ.get("OV_PRINTER") else [])


def main():
    from openvinyl.dsp import write_wav
    OUT.mkdir(parents=True, exist_ok=True)
    info = subprocess.run([sys.executable, "-m", "openvinyl", "info", "--diameter", str(DISC_D), "--outer", str(R_OUT),
                           "--inner", str(R_IN), "--rpm", str(RPM), "--amplitude", str(A_MAX_UM / 1000),
                           *base_args("", None)[2:]],
                          capture_output=True, text=True, cwd=REPO, env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")})
    g = dict(l.split(None, 1) for l in info.stdout.splitlines()[1:] if l.strip())
    pitch, avail = float(g["pitch_mm"]), float(g["audio_time_available_s"])
    seg, total, t_two = schedule(pitch)
    print(f"7 in at {RPM:.0f} RPM, pitch {pitch} mm, {avail} s of audio available; schedule {total:.2f} s")
    assert total <= avail, "schedule longer than the disc"
    x = signal(seg, total)
    wav = OUT / "test_disc.wav"
    write_wav(str(wav), x, FS)
    manifest = {"rpm": RPM, "pitch_mm": pitch, "a_max_um": A_MAX_UM, "lead_in_turns": 1.0, "fs": FS,
                "floor_layers": FLOOR_LAYERS, "two_layers_from_s": t_two,
                "sections": [{"kind": k, "t0": a, "t1": b, "f_hz": f, "a_um": am, "layers": nl,
                              "r_start_mm": round(radius(a, pitch), 1), "r_end_mm": round(radius(b, pitch), 1)}
                             for k, a, b, f, am, nl in seg]}
    (OUT / "test_disc.manifest.json").write_text(json.dumps(manifest, indent=1))
    for s in manifest["sections"]:
        lim = f"  (limit {a_limit_um(s['f_hz'], s['t1'], pitch) / MARGIN:5.1f} um)" if s["f_hz"] else ""
        print(f"  {s['t0']:5.2f}-{s['t1']:5.2f} s  r {s['r_start_mm']:4.1f}-{s['r_end_mm']:4.1f}  {s['kind']:16s} "
              f"{s['f_hz']:5.0f} Hz {s['a_um']:4.0f} um{lim}")
    base, base3mf = base_disc()
    out = OUT / "test_disc_7in_40rpm.gcode"
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, "-m", "openvinyl", "generate", str(wav), *base_args(base, base3mf),
                        "--diameter", str(DISC_D), "--outer", str(R_OUT), "--inner", str(R_IN), "--rpm", str(RPM),
                        "--amplitude", str(A_MAX_UM / 1000), "--no-dsp", "-o", str(out),
                        "--layers", "2", "--extra-layers-from", f"{t_two - 0.15:.3f}", "--floor-layers", str(FLOOR_LAYERS)],
                       capture_output=True, text=True, cwd=REPO, env=env)
    print(r.stdout[-3000:], r.stderr[-2000:])
    if r.returncode != 0:
        raise SystemExit("generate failed")
    rep = json.loads(out.with_suffix(".report.json").read_text())
    p = subprocess.run([sys.executable, "-m", "openvinyl", "preview", str(out)], capture_output=True, text=True,
                       cwd=REPO, env=env)
    print(p.stdout[-800:], p.stderr[-800:])
    # what the encoder's playback model (bead width, stylus) predicts each section returns, relative to the outer
    # reference: the analysis compares a real take against this, so the groove's own physics drop out
    import wave
    import analyze_test_disc as AT
    with wave.open(str(out.with_suffix(".preview.wav")), "rb") as w:
        pfs = w.getframerate()
        px = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767
    pred = AT.analyse(px, pfs, manifest, verbose=False)
    ref = next(r for r in pred["sections"] if r["kind"] == "reference outer")["mv_per_um"]
    for s_, r in zip(manifest["sections"], pred["sections"]):
        if "mv_per_um" in r:
            s_["predicted_rel_level"] = round(r["mv_per_um"] / ref, 4)
    (OUT / "test_disc.manifest.json").write_text(json.dumps(manifest, indent=1))
    print("predicted level per um vs the outer reference (encoder playback model): " +
          ", ".join(f"{s_['f_hz']:.0f} Hz {20 * math.log10(s_['predicted_rel_level']):+.0f} dB"
                    for s_ in manifest["sections"] if s_["kind"].startswith("tone")))
    print(f"wrote {out.name}{', its .gcode.3mf' if base3mf else ''}, the preview, test_disc.wav and the manifest in {OUT}")
    return rep


if __name__ == "__main__":
    main()
