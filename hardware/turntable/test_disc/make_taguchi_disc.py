"""Taguchi L9 groove test disc for the spine turntable (Nov 24 2026).

Four groove factors at three levels, in the nine runs of an L9 orthogonal array, printed as nine bands along one
7 in, 40 RPM spiral. Every band plays the same clip of the song (conditioned for its own radius and tracking limit)
over one continuous 75 Hz pilot, so the bands differ only in the groove. analyze_taguchi.py scores each band by the
error between what the pickup gives back (after restore.py) and what was cut, and picks the best level of each
factor from the main effects.

    A  bead spacing / bead width    1.00   1.05   1.10    (the crease the stylus sits in)
    B  bead layers per wall         1      2      3       (wall height)
    C  tracking limit, wall slope   5      7      9 deg   (how hard the music is cut; acceleration limit scales)
    D  floor layers under the band  0      1      2       (flat spiral floor, aligned under the crease)

The pitch stays that of the widest band (1.10), so the spiral is one continuous groove. Between bands, 0.75 s of
pilot-only groove: the bead spacing ramps across it, the floor steps a layer at a time, layers start or stop at its
middle. Radius is the noise factor (outer bands play faster, so cleaner): the band order is chosen so each factor's
levels sit at about the same mean radius, and --reverse prints the mirrored order for a second disc that cancels
what is left.

    python make_taguchi_disc.py SONG.wav [--start s] [--reverse] [--sim-only]
"""

import argparse
import itertools
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import make_test_disc as MT                      # noqa: E402  (disc size, base slicing)
import make_song_disc as SD                      # noqa: E402  (pickup simulation, score)
from openvinyl.dsp import read_wav, write_wav    # noqa: E402
from openvinyl.groove import GrooveParams, build_groove   # noqa: E402
from openvinyl.printer import get_profile        # noqa: E402
from openvinyl import song                       # noqa: E402

L9 = [(0, 0, 0, 0), (0, 1, 1, 1), (0, 2, 2, 2),
      (1, 0, 1, 2), (1, 1, 2, 0), (1, 2, 0, 1),
      (2, 0, 2, 1), (2, 1, 0, 2), (2, 2, 1, 0)]
FACTORS = {"A_spacing": [1.00, 1.05, 1.10], "B_layers": [1, 2, 3], "C_slope_deg": [5.0, 7.0, 9.0],
           "D_floor": [0, 1, 2]}
ACC_PER_DEG = 10.0            # m/s^2 of stylus acceleration allowed per degree of slope (7 deg -> 70, as v3)
A_MUSIC, A_PILOT = 0.040, 0.010
LEAD_IN_TURNS = 2
GAP_S = 0.75                  # pilot-only groove between bands, = the schedule's ramp


def band_order(reverse=False):
    """The run order along the disc (outer first). Each factor's three levels should sit at about the same mean
    position, so radius (the noise factor) does not pass for a factor's effect; adjacent bands should differ by at
    most one floor layer, so the floor steps stay short. Brute force over the 9! orders. Nine bands cannot balance
    all four columns of an L9: the best orders balance three and leave the fourth fully tied to radius. Spacing,
    slope and floor are balanced (to 0.7 of a band); layers goes 1 outer, 3 middle, 2 inner, and the mirrored
    disc (--reverse) undoes that, so the two discs together balance every factor exactly."""
    perms = np.array(list(itertools.permutations(range(9))), dtype=np.int8)     # perms[p, position] = run
    arr = np.array(L9)
    lev = arr[perms]                                                              # (P, position, factor)
    pos = np.arange(9)
    cost = np.zeros(len(perms))
    weight = (10.0, 1.0, 10.0, 10.0)                  # A, B, C, D: layers is the one left to the mirrored disc
    for f in range(4):
        for l in range(3):
            m = lev[:, :, f] == l
            cost += weight[f] * ((m * pos).sum(1) / 3.0 - 4.0) ** 2
    cost += 0.1 * (np.abs(np.diff(lev[:, :, 3], axis=1)) > 1).sum(1)
    cost += 1e-3 * (perms[:, 0] > perms[:, -1])                                  # of a mirrored pair, keep one
    i = int(np.argmin(cost))
    best = [int(r) for r in perms[i]]
    return (best[::-1] if reverse else best), float(cost[i])


def pick_clip(audio, fs, dur):
    """The stretch of the song with the most energy in the groove's band (150-800 Hz): the most for a band to say."""
    from scipy import signal
    sos = signal.butter(4, [150 / (fs / 2), 800 / (fs / 2)], "band", output="sos")
    b = signal.sosfiltfilt(sos, audio)
    n = int(dur * fs)
    c = np.concatenate([[0.0], np.cumsum(b * b)])
    e = c[n:] - c[:-n]                                   # energy of every window of n samples
    return int(np.argmax(e)) / fs


def geometry(amplitude, schedule):
    gp = GrooveParams(rpm=MT.RPM, record_diameter=MT.DISC_D, outer_groove_radius=MT.R_OUT,
                      inner_groove_radius=MT.R_IN, amplitude=amplitude, lead_in_turns=LEAD_IN_TURNS,
                      schedule=schedule, ramp_s=GAP_S)
    printer = get_profile(__import__("os").environ.get("OV_PRINTER", "p1s"))
    return gp, printer, gp.resolve(printer)


def layout(g, order):
    """Band audio windows [t0, t1) and the schedule (each band's end at the middle of the gap after it)."""
    total = g.audio_time_available - 0.05
    lb = (total - 8 * GAP_S) / 9
    bands, sched = [], []
    for k, run in enumerate(order):
        t0 = k * (lb + GAP_S)
        a, b, c, d = L9[run]
        lv = {"A_spacing": FACTORS["A_spacing"][a], "B_layers": FACTORS["B_layers"][b],
              "C_slope_deg": FACTORS["C_slope_deg"][c], "D_floor": FACTORS["D_floor"][d]}
        bands.append({"band": k + 1, "run": run + 1, "levels": [a + 1, b + 1, c + 1, d + 1], **lv,
                      "t0": round(t0, 4), "t1": round(t0 + lb, 4)})
        s = {"spacing": lv["A_spacing"], "layers": lv["B_layers"], "floor": lv["D_floor"]}
        if k < 8:
            s["t1"] = round(t0 + lb + GAP_S / 2, 4)
        sched.append(s)
    return bands, sched, total


def build_signal(clip, fs, g, bands, total):
    """The groove in mm: each band's clip conditioned for its radius and slope limit, one pilot under all."""
    n = int(total * fs)
    x = np.zeros(n)
    infos = []
    for b in bands:
        lb = b["t1"] - b["t0"]
        c = clip[: int(lb * fs)]
        y, info = song.prepare(c, fs, g, A_MUSIC, A_PILOT, slope_deg=b["C_slope_deg"],
                               acc_max_ms2=ACC_PER_DEG * b["C_slope_deg"], t0_audio=b["t0"], pilot=False)
        i0 = int(round(b["t0"] * fs))
        m = y * info["amplitude_mm"]
        x[i0:i0 + len(m)] += m[: n - i0]
        infos.append({k: info[k] for k in ("a_music_um", "a_music_rms_um", "slope_max_deg", "acc_max_ms2",
                                           "track_gain_mean_db", "lpf_outer_hz", "lpf_inner_hz")})
    tt = np.arange(n) / fs
    x += A_PILOT * np.sin(2 * math.pi * song.PILOT_HZ * tt) * np.clip(tt / 0.05, 0, 1) * np.clip((tt[-1] - tt) / 0.05, 0, 1)
    return x, infos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--start", type=float, default=None, help="clip start in the song, s (default: the busiest stretch)")
    ap.add_argument("--name", default="taguchi_l9")
    ap.add_argument("--reverse", action="store_true", help="mirror the band order (the second disc)")
    ap.add_argument("--sim-only", action="store_true", help="skip slicing and G-code")
    ap.add_argument("--plays", type=int, default=2, help="simulated plays for the dry run of the analysis")
    a = ap.parse_args()
    name = a.name + ("_rev" if a.reverse else "")
    out = REPO / "out" / "taguchi" / name
    out.mkdir(parents=True, exist_ok=True)

    order, cost = band_order(a.reverse)
    print("band order (runs, outer first):", [r + 1 for r in order], f" balance cost {cost:.3f}")
    audio, fs = read_wav(a.song)
    amp = A_MUSIC + A_PILOT
    for _ in range(2):                    # the pitch follows the real peak: condition, rebuild, condition again
        sched0 = [{"spacing": s, "layers": 1, "floor": 0} for s in FACTORS["A_spacing"]]
        gp, printer, g = geometry(amp, sched0)
        bands, sched, total = layout(g, order)
        lb = bands[0]["t1"] - bands[0]["t0"]
        start = pick_clip(audio, fs, lb) if a.start is None else a.start
        clip = audio[int(start * fs): int((start + lb) * fs)]
        gp, printer, g = geometry(amp, sched)
        x_mm, infos = build_signal(clip, fs, g, bands, total)
        amp = float(np.abs(x_mm).max())
    gp, printer, g = geometry(amp, sched)
    y = x_mm / amp
    print(f"{len(bands)} bands of {lb:.2f} s, gaps {GAP_S} s, clip {start:.2f}-{start + lb:.2f} s of the song; "
          f"peak {amp * 1000:.1f} um, pitch {g.pitch:.4f} mm")
    for b, inf in zip(bands, infos):
        b.update(inf)
        print(f"  band {b['band']}: run {b['run']} levels {b['levels']}  spacing {b['A_spacing']} layers "
              f"{b['B_layers']} slope {b['C_slope_deg']:g} floor {b['D_floor']}  music {inf['a_music_um']} um peak, "
              f"{inf['a_music_rms_um']} rms")

    wav = out / f"{name}.groove.wav"
    write_wav(str(wav), y, fs)
    (out / f"{name}.schedule.json").write_text(json.dumps(sched, indent=1))
    man = {"rpm": MT.RPM, "pitch_mm": g.pitch, "r_outer_mm": MT.R_OUT, "r_inner_mm": MT.R_IN,
           "lead_in_turns": LEAD_IN_TURNS, "segment_mm": gp.segment_length, "bead_width_mm": g.w_e,
           "source": Path(a.song).name, "clip_start_s": round(start, 3), "clip_s": round(lb, 4), "gap_s": GAP_S,
           "pilot_hz": song.PILOT_HZ, "a_pilot_um": A_PILOT * 1000, "amplitude_mm": round(amp, 5),
           "hpf_hz": 120.0, "audio_s": round(total, 4), "fs": fs, "factors": FACTORS, "l9": L9,
           "order": [r + 1 for r in order], "reverse": a.reverse, "bands": bands, "schedule": sched}
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))

    path = build_groove(y, fs, gp, printer)
    print("groove:", {k: path.stats[k] for k in ("slope_limited_fraction", "curvature_limited_fraction",
                                                  "max_slope_deg", "audio_time_s")})
    # dry run of the analysis on simulated plays (the pickup model knows nothing of the factors: this only checks
    # the pipeline end to end and how much radius alone moves the score)
    takes = [SD.simulate(path, seed=k)[0] for k in range(1, a.plays + 1)]
    tfs = 9615
    for k, tk in enumerate(takes):
        write_wav(str(out / f"{name}.sim{k + 1}.wav"), tk / (np.abs(tk).max() + 1e-9) * 0.89, tfs)
    if a.sim_only:
        return

    MT.OUT = out
    base, base3mf = MT.base_disc()
    gcode = out / f"{name}_7in_{MT.RPM:g}rpm.gcode"
    env = {**os.environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, "-m", "openvinyl", "generate", str(wav), *MT.base_args(base, base3mf),
                        "--diameter", str(MT.DISC_D), "--outer", str(MT.R_OUT), "--inner", str(MT.R_IN),
                        "--rpm", str(MT.RPM), "--amplitude", f"{amp:.5f}", "--no-dsp",
                        "--schedule", str(out / f"{name}.schedule.json"), "--ramp", str(GAP_S),
                        "-o", str(gcode), "--lead-in", str(LEAD_IN_TURNS)],
                       capture_output=True, text=True, cwd=REPO, env=env)
    print(r.stdout[-2500:], r.stderr[-1500:])
    if r.returncode != 0:
        raise SystemExit("generate failed")
    p = subprocess.run([sys.executable, "-m", "openvinyl", "preview", str(gcode)], capture_output=True, text=True,
                       cwd=REPO, env=env)
    print(p.stdout[-600:], p.stderr[-600:])
    print(f"wrote {gcode.name}" + (f" and {gcode.name}.3mf" if base3mf else "") + f" in {out}")


if __name__ == "__main__":
    main()
