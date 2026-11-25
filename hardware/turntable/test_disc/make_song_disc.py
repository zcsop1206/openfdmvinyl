"""
make_song_disc.py - a 7 in song record (40 or 45 RPM) for the spine turntable, conditioned by openvinyl.song (radius-following
low-pass, levelled and compressed, 40 um peaks, a 75 Hz pilot for the wow correction) on a spiral floor.

    python make_song_disc.py SONG.wav [--start 12.0] [--dur 33] [--rpm 45] [--name NAME] [--sim-only]

Writes out/song/<NAME>/: the conditioned signal, the manifest restore.py needs, the base disc sliced by Bambu Studio,
the spliced G-code and .gcode.3mf (print it from the printer's screen, not from Studio: see docs/BUILD.md), and
<NAME>.sim.wav / <NAME>.sim_restored.wav: what a take should sound like before and after the cleanup, from the
groove's own displacement through the measured pickup (about 5 mV per um, compressing past 30 um), a 0.8 % off-centre
wobble, mains hum, and take 5's real surface noise (the run-out, about 116 mV in the music band; the in-groove silences
of takes 4 and 5 measured 73 to 105 mV; during music takes 4 and 5 ran 107-177 mV, so this sits between them). <NAME>.sim_restored_<N>play.wav average N
simulated plays, each with fresh noise. On the v1 disc (no floor) the 130-600 Hz noise partly repeats between passes
(correlation ~0.5 between passes), so averaging gains less there than simulated; check two passes of the song disc.

After a take:   PYTHONPATH=src python -m openvinyl.restore TAKE.wav out/song/<NAME>/<NAME>.manifest.json -o clean.wav
"""
import argparse
import json
import math
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from scipy import signal as sg

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import make_test_disc as MT                      # noqa: E402  (disc size, base slicing)
from openvinyl.dsp import read_wav, write_wav    # noqa: E402
from openvinyl.groove import GrooveParams, build_groove   # noqa: E402
from openvinyl.printer import get_profile        # noqa: E402
from openvinyl import song, restore              # noqa: E402

A_MUSIC, A_PILOT = 0.040, 0.010
FLOOR_LAYERS = 1
GROOVE_LAYERS = 1                                # one bead per wall: landed at full level 5 of 11 times against 1 of 6 for two (Nov 24-25)
NOISE_TAKE = HERE / "data" / "groove_noise.wav"   # 9.0 s of a real take (take 5, Nov 22) past the test disc's groove end
NOISE_SPAN = (0.0, 6 * 60 / 39.8)               # (no-floor disc), six whole turns at 39.8 RPM: its loudness
                                                 # pulses with the turn (infill crossings), so it is tiled by whole turns
LEAD_IN_TURNS = 2                                # one whole turn of plain groove for the noise example, after the drop


BEAD_SPACING = 1.1                               # bead centre spacing / bead width (set by --spacing)


def geometry(amplitude):
    gp = GrooveParams(rpm=MT.RPM, record_diameter=MT.DISC_D, outer_groove_radius=MT.R_OUT,
                      inner_groove_radius=MT.R_IN, amplitude=amplitude, floor_layers=FLOOR_LAYERS,
                      groove_layers=GROOVE_LAYERS, bead_spacing_factor=BEAD_SPACING,
                      lead_in_turns=LEAD_IN_TURNS)
    printer = get_profile(__import__("os").environ.get("OV_PRINTER", "p1s"))
    return gp, printer, gp.resolve(printer)


def simulate(path, fs_out=9615, mv_per_um=5.0, sat_um=30.0, wow_pct=0.8, speed=0.995, seed=1):
    """A take as the pickup would give it, in volts at the Arduino's rate."""
    g = path.geometry
    t_play = np.arange(0, path.t[-1], 1 / 44100)
    x_um = np.interp(t_play, path.t, path.x) * 1000
    y = mv_per_um * 1e-3 * sat_um * np.tanh(x_um / sat_um)            # volts
    # playback clock: off-centre wobble growing as 1/r, and the platter a little slow
    W = 2 * math.pi * MT.RPM / 60 * speed
    t_rec = np.arange(0, t_play[-1] / speed - 0.5, 1 / fs_out)
    r = np.interp(t_rec * speed, path.t, path.r_spiral)
    rate = speed * (1 + wow_pct / 100 * (r[0] / r) * np.sin(W * t_rec + 1.0))
    tau = np.concatenate([[0], np.cumsum(rate[1:] / fs_out)])
    yr = np.interp(tau, t_play, y)
    # noise: take 5's run-out, tiled; plus 60 Hz and 180 Hz hum
    with wave.open(str(NOISE_TAKE)) as w:
        nf = w.getframerate()
        nx = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767 * 2.5
    nz = nx[int(NOISE_SPAN[0] * nf):int(NOISE_SPAN[1] * nf)]
    nz = nz - nz.mean()
    rng = np.random.default_rng(seed)
    reps = int(math.ceil(len(yr) / len(nz))) + 1
    # the noise's loudness stays locked to the disc angle, as on a real disc, but its waveform is new on every play
    # (take 5's passes correlate at 0.07): keep the STFT magnitudes of whole turns, draw fresh phases
    turn = len(nz) / 6.0
    tiles = []
    for _ in range(reps):
        f_, t_, Z = sg.stft(np.roll(nz, int(round(rng.integers(6) * turn))), nf, nperseg=256)
        Z = np.abs(Z) * np.exp(2j * np.pi * rng.random(Z.shape))
        tiles.append(sg.istft(Z, nf, nperseg=256)[1][: len(nz)])
    noise = np.concatenate(tiles)[: len(yr)]
    tt = np.arange(len(yr)) / fs_out
    hum = 0.03 * np.sin(2 * math.pi * 60 * tt) + 0.02 * np.sin(2 * math.pi * 180 * tt + 0.3)
    return yr + noise + hum, fs_out


def score(clean_ref, fs_ref, test, fs_test, man):
    """SNR (dB) of a take or restored take against the groove signal, after best alignment and gain, in the music
    band. A crude number, but it moves the right way."""
    from fractions import Fraction
    fr = Fraction(fs_test, fs_ref).limit_denominator(4000)
    ref = sg.resample_poly(clean_ref, fr.numerator, fr.denominator)
    sos = sg.butter(4, [110 / (fs_test / 2), 900 / (fs_test / 2)], "band", output="sos")
    ref, tst = sg.sosfiltfilt(sos, ref), sg.sosfiltfilt(sos, test)
    n = min(len(ref), len(tst))
    c = sg.correlate(tst[:n], ref[:n], mode="full", method="fft")
    lag = int(np.argmax(np.abs(c))) - (n - 1)
    if lag >= 0:
        a, b = ref[: n - lag], tst[lag:n]
    else:
        a, b = ref[-lag:n], tst[: n + lag]
    # SNR over 3 s blocks (re-aligned per block, so leftover wow counts against the take only beyond a block)
    out = []
    blk = 3 * fs_test
    for i in range(0, len(a) - blk, blk):
        aa, bb = a[i:i + blk], b[i:i + blk]
        cc = sg.correlate(bb, aa, mode="same", method="fft")
        l = int(np.argmax(np.abs(cc))) - len(aa) // 2
        aa = np.roll(aa, l)
        k = np.dot(bb, aa) / np.dot(aa, aa)
        out.append(10 * np.log10(np.sum((k * aa) ** 2) / np.sum((bb - k * aa) ** 2)))
    return float(np.median(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("song")
    ap.add_argument("--start", type=float, default=0.0, help="clip start in the song, s")
    ap.add_argument("--dur", type=float, default=None, help="clip length, s (default: what the disc holds)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--rpm", type=float, default=MT.RPM, help="platter speed (45 gives 12 %% more bandwidth and 27 %% "
                    "more bend budget for 11 %% less time; the servo test held 45.00 RPM, and tops out near 48)")
    ap.add_argument("--bass", type=float, default=0.6, help="virtual-bass gain (harmonics of the 40-120 Hz band), 0 = off")
    ap.add_argument("--spacing", type=float, default=1.1, help="bead centre spacing / bead width (1.05: the crease "
                    "stays under a 0.1 mm tip even if the beads print 0.07 mm narrow)")
    ap.add_argument("--slope", type=float, default=7.0, help="tracking limit: steepest wall slope, degrees")
    ap.add_argument("--acc", type=float, default=70.0, help="tracking limit: stylus sideways acceleration, m/s^2")
    ap.add_argument("--plays", default="1,4,8", help="simulate the cleanup averaging this many plays")
    ap.add_argument("--sim-only", action="store_true", help="skip slicing and G-code")
    a = ap.parse_args()
    MT.RPM = a.rpm
    global BEAD_SPACING
    BEAD_SPACING = a.spacing
    a.plays = [int(v) for v in a.plays.split(",")]
    name = a.name or Path(a.song).stem.replace(" ", "_")
    out = REPO / "out" / "song" / name
    out.mkdir(parents=True, exist_ok=True)

    gp, printer, g = geometry(A_MUSIC + A_PILOT)
    audio, fs = read_wav(a.song)
    dur = min(a.dur or g.audio_time_available, g.audio_time_available - 0.05)
    clip = audio[int(a.start * fs):int((a.start + dur) * fs)]
    print(f"7 in at {MT.RPM:.0f} RPM: {g.audio_time_available:.1f} s of audio; clip {a.start:.1f}-{a.start + len(clip) / fs:.1f} s")
    y, info = song.prepare(clip, fs, g, A_MUSIC, A_PILOT, bass=a.bass, slope_deg=a.slope, acc_max_ms2=a.acc)
    # the tracking limit leaves the peak under A_MUSIC + A_PILOT; the pitch follows the amplitude, so rebuild the
    # geometry at the real peak and condition once more against the radii it gives
    gp, printer, g = geometry(info["amplitude_mm"])
    y, info = song.prepare(clip, fs, g, A_MUSIC, A_PILOT, bass=a.bass, slope_deg=a.slope, acc_max_ms2=a.acc)
    gp, printer, g = geometry(info["amplitude_mm"])
    print("conditioned:", info)
    wav = out / f"{name}.groove.wav"
    write_wav(str(wav), y, fs)
    man = {"rpm": MT.RPM, "pitch_mm": g.pitch, "r_outer_mm": MT.R_OUT, "r_inner_mm": MT.R_IN,
           "lead_in_turns": gp.lead_in_turns, "segment_mm": gp.segment_length, "bead_width_mm": g.w_e,
           "floor_layers": FLOOR_LAYERS, "groove_layers": GROOVE_LAYERS, "bead_spacing_factor": BEAD_SPACING, "source": Path(a.song).name, "clip_start_s": a.start, **info}
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))

    # the groove the encoder will print (the same build_groove call generate makes), played through the pickup model
    path = build_groove(y, fs, gp, printer)
    print("groove:", {k: path.stats[k] for k in ("slope_limited_fraction", "curvature_limited_fraction",
                                                  "max_slope_deg", "audio_time_s")})
    take, tfs = simulate(path)
    write_wav(str(out / f"{name}.sim.wav"), take / (np.abs(take).max() + 1e-9) * 0.89, tfs)
    ref = np.interp(np.arange(0, path.t[-1], 1 / fs), path.t, path.x)
    s_raw = score(ref, fs, take, tfs, man)
    man["sim"] = {"snr_raw_db": round(s_raw, 1)}
    print(f"simulated take: SNR {s_raw:+.1f} dB raw (music band, against the groove)")
    takes = [take] + [simulate(path, seed=k)[0] for k in range(2, max(a.plays) + 1)]
    for n in a.plays:
        clean, rinfo = restore.restore(takes[:n], tfs, man, verbose=False)
        s_n = score(ref, fs, clean, tfs, man)
        write_wav(str(out / f"{name}.sim_restored_{n}play.wav"), clean, tfs)
        man["sim"][f"snr_restored_{n}play_db"] = round(s_n, 1)
        print(f"  restored from {n} play(s): SNR {s_n:+.1f} dB   wow found "
              f"{rinfo['per_pass'][0].get('wow_once_pct')} %, speed {rinfo['per_pass'][0].get('speed_error_pct')} %")
    (out / f"{name}.manifest.json").write_text(json.dumps(man, indent=1))
    if a.sim_only:
        return

    MT.OUT = out
    base, base3mf = MT.base_disc()
    gcode = out / f"{name}_7in_{MT.RPM:g}rpm.gcode"
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO / "src")}
    r = subprocess.run([sys.executable, "-m", "openvinyl", "generate", str(wav), *MT.base_args(base, base3mf),
                        "--diameter", str(MT.DISC_D), "--outer", str(MT.R_OUT), "--inner", str(MT.R_IN),
                        "--rpm", str(MT.RPM), "--amplitude", f"{info['amplitude_mm']:.5f}", "--no-dsp", "--layers", str(GROOVE_LAYERS), "--bead-spacing", str(BEAD_SPACING),
                        "-o", str(gcode), "--floor-layers", str(FLOOR_LAYERS),
                        "--lead-in", str(LEAD_IN_TURNS)],
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
