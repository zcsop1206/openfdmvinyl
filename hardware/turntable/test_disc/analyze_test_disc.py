"""
analyze_test_disc.py - read a recording of the test disc (make_test_disc.py) and report speed, wow and flutter,
frequency response, distortion, tracking and noise.

    python analyze_test_disc.py take1.wav [--manifest PATH]   # the raw WAV from adc_capture/capture.py
    python analyze_test_disc.py --synthetic                    # self-test: a fake take with known faults

capture.py writes take1.wav as AC volts / 2.5 at 9615 Hz (and take1.hp.wav, filtered and normalised: not this
one). Start the recording, then lower the stylus anywhere in the lead-in or the first silence: the script finds
the outer 333 Hz reference by itself, takes the speed from its pitch, and lays the manifest's sections over the
take with that speed error accounted for. Each section is read in its middle (0.2 s off each end).

What each number is:
  speed         333 Hz reference pitch vs 330: the platter's speed error, %
  wow, flutter  the reference's pitch wobble: peak-to-peak and RMS, %, and the strongest wobble rate (0.67 Hz =
                once per turn: an off-centre hole or spindle)
  level         the fitted fundamental, mV, and mV per um of commanded groove amplitude
  response      level per um against frequency, fitted with the cartridge's first-order high-pass (its capacitance
                into the 500 kOhm bias): the corner fc, and how far each tone falls below that model (nozzle smear
                and stylus size at the high tones)
  THD           harmonics 2-5 against the fundamental, % (mains harmonics of 60 Hz are fitted out separately)
  tracking      the 110 Hz ladder: level should double with amplitude and THD stay flat; the first step where THD
                jumps (x3 or over 10 %) or the level stops growing is where the stylus stops following
  noise         RMS in the silences, 80-2000 Hz, and as um of groove at the reference's mV/um
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, hilbert, sosfiltfilt

HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE.parents[2] / "out" / "testdisc" / "test_disc.manifest.json"
FS_UNO = 16_000_000 / 128 / 13
MAINS = 60.0


def read_take(path):
    import wave
    with wave.open(str(path), "rb") as w:
        fs = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767.0
    if abs(fs - FS_UNO) < 1:                 # capture.py rounds the rate in the header
        fs = FS_UNO
    return x * 2.5, fs                       # volts


def bandpass(x, fs, lo, hi, order=4):
    return sosfiltfilt(butter(order, [lo, hi], "bandpass", fs=fs, output="sos"), x)


def find_reference(x, fs, f_ref, min_s=1.5):
    """Start of the first sustained tone near f_ref after the stylus lands (time, s)."""
    y = bandpass(x, fs, f_ref * 0.85, f_ref * 1.15)
    env = np.abs(hilbert(y))
    w = int(0.05 * fs)
    env = np.convolve(env, np.ones(w) / w, mode="same")
    thr = 0.3 * np.percentile(env, 99)
    on = env > thr
    n = int(min_s * fs)
    i = 0
    while i < len(on) - n:
        if on[i] and on[i:i + n].mean() > 0.95:
            return i / fs
        i += 1
    raise SystemExit("could not find the 333 Hz reference: is the take the raw capture.py WAV, and the stylus down?")


def tone_phase(seg, fs, f):
    """The tone's own instantaneous phase (rad): a fit locked to it is not smeared by wow."""
    y = bandpass(seg, fs, f * 0.8, f * 1.2)
    return np.unwrap(np.angle(hilbert(y)))


def fit_tone(seg, fs, f, n_harm=5, mains=True):
    """Least squares on [sin k phi, cos k phi] with phi the tone's measured phase (so speed wobble does not smear
    it), plus the mains harmonics at their fixed frequencies: amplitudes (V peak) and the residual RMS."""
    t = np.arange(len(seg)) / fs
    phi = tone_phase(seg, fs, f)
    e = int(0.01 * fs)                                       # drop the band-pass edges
    seg, t, phi = seg[e:-e], t[e:-e], phi[e:-e]
    cols, tags = [np.ones_like(t)], ["dc"]
    for k in range(1, n_harm + 1):
        if k * f < fs / 2 * 0.9:
            cols += [np.sin(k * phi), np.cos(k * phi)]
            tags += [("h", k)] * 2
    if mains:
        for m in range(1, int(2000 / MAINS) + 1):
            fm = m * MAINS
            if all(abs(fm - k * f) > 1.5 for k in range(1, n_harm + 1)):     # ~1 s windows resolve 1.5 Hz
                cols += [np.sin(2 * math.pi * fm * t), np.cos(2 * math.pi * fm * t)]
                tags += [("m", m)] * 2
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
    amp = {}
    for i in range(1, len(tags), 2):
        amp[tags[i]] = math.hypot(coef[i], coef[i + 1])
    resid = seg - A @ coef
    return amp, float(np.sqrt(np.mean(resid ** 2)))


def peak_freq(seg, fs, f_guess):
    """Frequency of the strongest line near f_guess (zero-padded FFT, parabolic peak)."""
    n = 1 << int(math.ceil(math.log2(len(seg) * 8)))
    s = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    fr = np.fft.rfftfreq(n, 1 / fs)
    band = (fr > f_guess * 0.9) & (fr < f_guess * 1.1)
    i = np.flatnonzero(band)[np.argmax(s[band])]
    a, b, c = np.log(s[i - 1:i + 2] + 1e-30)
    p = 0.5 * (a - c) / (a - 2 * b + c)
    return fr[i] + p * (fr[1] - fr[0])


def wow(seg, fs, f0, f_turn):
    """Instantaneous frequency of a tone: mean, peak-to-peak and RMS deviation (%), and the once-per-turn part
    (fitted at the turn rate: a 3 s window holds only two turns, too few for a spectrum)."""
    y = bandpass(seg, fs, f0 * 0.8, f0 * 1.2)
    ph = np.unwrap(np.angle(hilbert(y)))
    fi = np.gradient(ph) * fs / (2 * math.pi)
    k = int(0.15 * fs)
    fi = fi[k:-k]
    w = int(0.02 * fs)
    fi = np.convolve(fi, np.ones(w) / w, mode="valid")           # 20 ms: drop the carrier-rate ripple
    mean = fi.mean()
    dev = fi / mean - 1
    t = np.arange(len(dev)) / fs
    A = np.stack([np.ones_like(t), t, np.sin(2 * math.pi * f_turn * t), np.cos(2 * math.pi * f_turn * t)], axis=1)
    c, *_ = np.linalg.lstsq(A, dev, rcond=None)
    turn_pp = 2 * math.hypot(c[2], c[3])
    rest = dev - A @ c
    return mean, 100 * (dev.max() - dev.min()), 100 * dev.std(), 100 * turn_pp, 100 * (rest.max() - rest.min())


def analyse(x, fs, man, verbose=True):
    secs = man["sections"]
    ref = next(s for s in secs if s["kind"] == "reference outer")
    t_on = find_reference(x, fs, ref["f_hz"])
    # speed from the reference's pitch, then lay the schedule over the take
    i0, i1 = int((t_on + 0.3) * fs), int((t_on + (ref["t1"] - ref["t0"]) * 0.97 - 0.3) * fs)
    f_ref = wow(x[i0:i1], fs, peak_freq(x[i0:i1], fs, ref["f_hz"]), man["rpm"] / 60.0)[0]
    k = f_ref / ref["f_hz"]                                      # >1: the platter runs fast
    to_rec = lambda t: t_on + (t - ref["t0"]) / k

    def cut(s, trim=0.2):
        a, b = to_rec(s["t0"]) + trim, to_rec(s["t1"]) - trim
        return x[max(0, int(a * fs)):min(len(x), int(b * fs))]
    out = {"speed_error_pct": 100 * (k - 1), "f_ref_hz": f_ref, "reference_onset_s": t_on, "sections": []}
    # reference: wow and flutter
    rows = []
    for s in secs:
        seg = cut(s)
        row = {"kind": s["kind"], "f_hz": s["f_hz"], "a_um": s["a_um"], "r_mm": (s["r_start_mm"] + s["r_end_mm"]) / 2,
               "pred": s.get("predicted_rel_level"), "layers": s.get("layers", 1)}
        if len(seg) < fs * 0.3:
            row["note"] = "not in the take"
            rows.append(row)
            continue
        if s["f_hz"]:
            f = s["f_hz"] * k
            amp, resid = fit_tone(seg, fs, f)
            h1 = amp.get(("h", 1), 0.0)
            harm = [amp.get(("h", n), 0.0) for n in range(2, 6)]
            row.update(level_mv=1000 * h1, mv_per_um=1000 * h1 / s["a_um"], _harm=[h1] + harm, _f=f,
                       thd_pct=100 * math.sqrt(sum(h * h for h in harm)) / h1 if h1 > 0 else float("nan"),
                       h2_pct=100 * harm[0] / h1 if h1 else float("nan"), h3_pct=100 * harm[1] / h1 if h1 else float("nan"),
                       hum_mv=1000 * math.sqrt(sum(v * v for kk, v in amp.items() if kk[0] == "m")),
                       resid_mv=1000 * resid, clipped=bool(np.abs(seg).max() > 2.45))
            if s["kind"].startswith("reference"):
                m, pp, rms, turn, rest = wow(seg, fs, f, k * man["rpm"] / 60.0)
                row.update(f_meas_hz=m, wow_pp_pct=pp, wow_rms_pct=rms, once_per_turn_pp_pct=turn, rest_pp_pct=rest)
        else:
            y = bandpass(seg, fs, 80, min(2000, fs / 2 * 0.9))
            row.update(noise_mv_rms=1000 * float(np.sqrt(np.mean(y ** 2))))
        rows.append(row)
    out["sections"] = rows
    ref_row = next(r for r in rows if r["kind"] == "reference outer")
    s_ref = ref_row["mv_per_um"]
    # frequency response: divide out what the groove itself does (the encoder's predicted level per section, if the
    # manifest has it), then fit the cartridge's first-order high-pass on what is left
    tones = [r for r in rows if r["kind"].startswith("tone") and "mv_per_um" in r and r["layers"] == 1]
    for r in rows:
        if "mv_per_um" in r and r.get("pred"):
            r["vs_predicted_db"] = 20 * math.log10((r["mv_per_um"] / s_ref) / r["pred"])
    use = [r for r in tones if r.get("pred", 1) and r.get("pred", 1) > 0.1]       # skip tones the groove all but erases
    fs_ = np.array([r["f_hz"] for r in use])
    g = np.array([r["mv_per_um"] / (r["pred"] if r.get("pred") else 1.0) for r in use])
    best = None
    for fc in np.arange(50.0, 1200.0, 2.0):
        sel = fs_ <= (800 if any(r.get("pred") for r in use) else 400)
        h = fs_[sel] / np.sqrt(fs_[sel] ** 2 + fc ** 2)
        s0 = (g[sel] @ h) / (h @ h)
        err = np.sum((g[sel] - s0 * h) ** 2)
        if best is None or err < best[0]:
            best = (err, fc, s0)
    _, fc, s0 = best
    for r in tones:
        model = s0 * r["f_hz"] / math.hypot(r["f_hz"], fc) * (r["pred"] if r.get("pred") else 1.0)
        r["vs_model_db"] = 20 * math.log10(r["mv_per_um"] / model)
        r["vs_ref_db"] = 20 * math.log10(r["mv_per_um"] / s_ref)
    out["highpass_fc_hz"] = fc
    # THD referred to the groove: undo the high-pass, which passes the harmonics more than the fundamental
    hp = lambda f_: f_ / math.hypot(f_, fc)
    for r in rows:
        if "_harm" in r:
            h = r.pop("_harm"); f_ = r.pop("_f")
            g1 = h[0] / hp(f_)
            r["thd_groove_pct"] = 100 * math.sqrt(sum((hn / hp((n + 2) * f_)) ** 2 for n, hn in enumerate(h[1:]))) / g1 if g1 else float("nan")
    out["sensitivity_mv_per_um_flat"] = s0
    # tracking: the ladder
    # one ladder per groove layer count (Nov 22 disc: one bead layer on the outer half, two on the inner)
    out["mistrack_by_layers"] = {}
    for n_l in sorted({r["layers"] for r in rows if r["kind"].startswith("ladder")}):
        lad = [r for r in rows if r["kind"].startswith("ladder") and "thd_pct" in r and r["layers"] == n_l]
        base_thd = lad[0]["thd_groove_pct"] if lad else float("nan")
        m = None
        for a, b in zip(lad, lad[1:]):
            growth = (b["level_mv"] / a["level_mv"]) / (b["a_um"] / a["a_um"])
            b["level_growth"] = growth
            if m is None and (b["thd_groove_pct"] > max(3 * base_thd, 10.0) or growth < 0.8 or b["clipped"]):
                m = b["a_um"]
        out["mistrack_by_layers"][n_l] = (m, max((r["a_um"] for r in lad), default=0))
    out["mistrack_from_um"] = out["mistrack_by_layers"].get(1, (None, 0))[0]
    # noise
    sil = [r for r in rows if "noise_mv_rms" in r]
    for r in sil:
        r["noise_um_rms"] = r["noise_mv_rms"] / s_ref
    if verbose:
        report(out)
    return out


def report(o):
    print(f"speed: reference at {o['f_ref_hz']:.2f} Hz -> platter {o['speed_error_pct']:+.2f} % "
          f"({40 * (1 + o['speed_error_pct'] / 100):.2f} RPM)")
    for r in o["sections"]:
        if "wow_pp_pct" in r:
            print(f"{r['kind']:16s} (r {r['r_mm']:.0f}): wow and flutter {r['wow_pp_pct']:.2f} % peak-to-peak "
                  f"({r['wow_rms_pct']:.2f} % RMS): once per turn {r['once_per_turn_pp_pct']:.2f} % p-p (an off-centre "
                  f"hole or spindle), the rest {r['rest_pp_pct']:.2f} % p-p (servo, drag)")
    print(f"frequency response (cartridge high-pass fitted: fc {o['highpass_fc_hz']:.0f} Hz, "
          f"{o['sensitivity_mv_per_um_flat']:.1f} mV/um above it):")
    print("    section           f Hz   um    mV   mV/um  vs ref dB  vs predicted dB  vs model dB  THD % (groove)   hum mV")
    for r in o["sections"]:
        if "mv_per_um" in r:
            print(f"    {r['kind']:16s} {r['f_hz']:5.0f} {r['a_um']:4.0f} {r['level_mv']:6.1f} {r['mv_per_um']:6.2f} "
                  f"{r.get('vs_ref_db', float('nan')):9.1f} {r.get('vs_predicted_db', float('nan')):15.1f} {r.get('vs_model_db', float('nan')):11.1f} "
                  + (f"{r['thd_pct']:6.2f} ({r['thd_groove_pct']:5.2f})" if r["level_mv"] > 4 * r["resid_mv"] else "  near noise   ")
                  + f" {r['hum_mv']:7.2f}" + ("  CLIPPED" if r["clipped"] else ""))
    for n_l, (m, top) in o["mistrack_by_layers"].items():
        print(f"tracking (110 Hz ladder, {n_l} bead layer{'s' if n_l > 1 else ''}): "
              f"{f'clean to {top:.0f} um' if m is None else f'mistracks from {m:.0f} um'}")
    for r in o["sections"]:
        if "noise_mv_rms" in r:
            print(f"noise, {r['kind']:14s}: {r['noise_mv_rms']:.2f} mV RMS (80-2000 Hz) = {r['noise_um_rms']:.2f} um of groove")


def figure(o, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2), dpi=110)
    tones = [r for r in o["sections"] if r["kind"].startswith("tone") and "mv_per_um" in r]
    f = np.array([r["f_hz"] for r in tones]); g = np.array([r["mv_per_um"] for r in tones])
    fc, s0 = o["highpass_fc_hz"], o["sensitivity_mv_per_um_flat"]
    ff = np.geomspace(80, 800, 200)
    ax[0].semilogx(ff, 20 * np.log10(s0 * ff / np.hypot(ff, fc)), color="#888", lw=1, label=f"cartridge high-pass, fc {fc:.0f} Hz")
    ax[0].semilogx(f, 20 * np.log10(g), "o-", color="#c0392b", label="measured (per um of groove)")
    ax[0].set_xlabel("Hz"); ax[0].set_ylabel("dB re 1 mV/um"); ax[0].grid(alpha=0.3, which="both"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"frequency response; speed {o['speed_error_pct']:+.2f} %", fontsize=9)
    for n_l, col in ((1, "#2b2b29"), (2, "#2e86c1")):
        lad = [r for r in o["sections"] if r["kind"].startswith("ladder") and "thd_groove_pct" in r and r["layers"] == n_l]
        if lad:
            ax[1].plot([r["a_um"] for r in lad], [r["thd_groove_pct"] for r in lad], "o-", color=col,
                       label=f"{n_l} bead layer{'s' if n_l > 1 else ''}")
    ax[1].set_xlabel("groove amplitude, um (110 Hz)"); ax[1].set_ylabel("THD, % (groove)"); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=8)
    ax[1].set_title("tracking ladders: " + ", ".join(f"{n} layer: " + ("clean" if m is None else f"from {m:.0f} um")
                                                    for n, (m, _) in o["mistrack_by_layers"].items()), fontsize=9)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")


def synthetic(man, seed=1):
    """A fake take with known faults: +0.6 % speed, 0.4 % p-p once-per-turn wow, the cartridge's 318 Hz high-pass,
    45 mV/um, smear above 450 Hz, 2 % third harmonic from 60 um and 15 % at 80 um (mistracking), 60 Hz hum and
    its harmonics, ADC noise and 10-bit steps."""
    rng = np.random.default_rng(seed)
    fs = FS_UNO
    k, wpp = 1.006, 0.004
    lead = 2.3                                   # stylus lands 2.3 s before the schedule's t = 0
    total = man["sections"][-1]["t1"] / k + lead + 1.0
    t = np.arange(int(total * fs)) / fs
    # play time of the groove at each take sample: speed k with once-per-turn wow
    tau = (t - lead) * k + (wpp / 2) / (2 * math.pi * 40 / 60) * np.sin(2 * math.pi * (40 / 60) * k * t) * k
    x = np.zeros_like(t)
    for s in man["sections"]:
        if not s["f_hz"]:
            continue
        on = (tau >= s["t0"]) & (tau < s["t1"])
        ph = 2 * math.pi * s["f_hz"] * (tau[on] - s["t0"])
        a = s["a_um"] * 0.045                     # volts at the stylus, 45 mV/um
        h3 = 0.15 if s["a_um"] >= 80 else 0.02 if s["a_um"] >= 60 else 0.002
        # the groove's own response: the encoder's prediction where the manifest has it (relative to the outer
        # reference, which is 1), else a plain smear roll-off
        p_max = max((q.get("predicted_rel_level") or 0) for q in man["sections"])
        smear = (s["predicted_rel_level"] / p_max if s.get("predicted_rel_level") and p_max   # low tones ~ full level
                 else 1.0 / math.sqrt(1 + (s["f_hz"] / 650.0) ** 6))
        x[on] += a * smear * (np.sin(ph) + h3 * np.sin(3 * ph))
    from scipy.signal import sosfilt
    x = sosfilt(butter(1, 318, "highpass", fs=fs, output="sos"), x)         # one first-order pass, like the cartridge
    for m, a in ((1, 0.004), (2, 0.002), (3, 0.0015), (5, 0.001)):
        x += a * np.sin(2 * math.pi * MAINS * m * t + rng.uniform(0, 6.3))
    x += rng.normal(0, 0.0015, len(t))
    x = np.round((x + 2.5) * 1023 / 5) * 5 / 1023 - 2.5
    x[t < lead - 0.5] = 0.0                      # stylus not down yet
    return x, fs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take", nargs="?")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--synthetic", action="store_true")
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    if a.synthetic:
        x, fs = synthetic(man)
        print("synthetic take: +0.60 % speed, 0.40 % p-p wow once per turn, high-pass 318 Hz, 45 mV/um, the groove "
              "response the encoder predicts, 2 % H3 at 60 um and 15 % at 80 um (mistracking)")
    else:
        if not a.take:
            ap.error("give the raw take WAV (or --synthetic)")
        x, fs = read_take(a.take)
    o = analyse(x, fs, man)
    png = Path(a.take).with_suffix(".analysis.png") if a.take else DEFAULT_MANIFEST.parent / "synthetic.analysis.png"
    figure(o, png)
    print(f"wrote {png.name}")
    if a.take:
        p = Path(a.take).with_suffix(".analysis.json")
        p.write_text(json.dumps(o, indent=1, default=float))
        print(f"wrote {p.name}")
    return o


if __name__ == "__main__":
    main()
