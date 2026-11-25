"""Score a Taguchi L9 groove disc (make_taguchi_disc.py) from its take(s) and pick the best level of each factor.

    python analyze_taguchi.py MANIFEST.json TAKE.wav [TAKE2.wav ...] [--out DIR]

Each take is cleaned with restore.py (hum, clicks, pitch from the pilot, the music band; no averaging and no noise
reduction, so the score is the groove's, not the cleanup's). Each play of the disc is cut into its nine bands on the
disc's clock and every band is scored against what was cut there:

  error    SNR, dB: the band's restored audio against its commanded groove signal after the best delay and gain,
           in 110-900 Hz, over the band minus 0.35 s at each end (the clip's fades). Larger is better; this is
           already Taguchi's larger-is-better S/N, so it is used as is.
  pilot    mean level (mV) and the depth of its 2.8 Hz pulsing (%), from the raw take: how well the band tracked.

With several plays (or takes), each band's SNR is the mean over plays. Main effects: for each factor, the mean SNR
of the three runs at each level; the best level is the highest, and the factor's range (best - worst) ranks how much
it matters. The predicted SNR of the best combination is the grand mean plus each factor's (best - mean).
Writes taguchi_report.json and taguchi_main_effects.png next to the manifest (or in --out).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))

from openvinyl import restore                     # noqa: E402

TRIM_S = 0.35


def load_take(path):
    fs, x = wavfile.read(path)
    x = x.astype(float)
    if x.ndim > 1:
        x = x[:, 0]
    if np.abs(x).max() > 2.0:                     # capture.py int16: volts = int16 / 32767 * 2.5
        x = x / 32767 * 2.5
    return fs, x - x.mean()


def plays_of(x, fs, man):
    """Each needle drop of the take on the disc's clock: [(audio on the disc clock, pilot level, pilot pulsing,
    raw slice start, speed)]. Starts at the pilot's onset, which is audio time 0 (the pilot runs the whole disc)."""
    x = restore.notch_hum(x, fs)
    x, _, _ = restore.declick(x, fs)
    found, cyc, amp = restore.find_passes(x, fs, man["pilot_hz"])
    out = []
    for i0, i1 in found:
        y, _, info = restore.restore_pass(x, fs, man, i0, i1, cyc, amp, ("wow", "band"))
        out.append({"audio": y, "i0": i0, "i1": i1, "speed": 1 + info.get("speed_error_pct", 0) / 100,
                    "info": info, "x": x})
    return out


def band_metrics(play, fs, man, ref, rfs):
    """Per band: SNR against the commanded groove, pilot level and its 2.8 Hz pulsing."""
    y = play["audio"]
    sos = signal.butter(4, [110 / (fs / 2), 900 / (fs / 2)], "band", output="sos")
    yb = signal.sosfiltfilt(sos, y)
    r = signal.resample_poly(ref, fs, rfs) if rfs != fs else ref
    rb = signal.sosfiltfilt(sos, r)
    # one delay for the whole play first (the pilot onset is only good to ~50 ms)
    n = min(len(yb), len(rb))
    c = signal.correlate(yb[:n], rb[:n], mode="full", method="fft")
    lag0 = int(np.argmax(np.abs(c))) - (n - 1)
    cyc, amp = restore.pilot_phase(play["x"], fs, man["pilot_hz"], bw=8.0)
    res = []
    for b in man["bands"]:
        a0, a1 = int((b["t0"] + TRIM_S) * fs), int((b["t1"] - TRIM_S) * fs)
        seg_r = rb[a0:a1]
        best = None
        for d in range(-int(0.03 * fs), int(0.03 * fs) + 1, 1):      # fine delay per band
            j0 = a0 + lag0 + d
            if j0 < 0 or j0 + len(seg_r) > len(yb):
                continue
            seg_y = yb[j0:j0 + len(seg_r)]
            k = np.dot(seg_y, seg_r) / (np.dot(seg_r, seg_r) + 1e-18)
            e = seg_y - k * seg_r
            snr = 10 * math.log10(np.sum((k * seg_r) ** 2) / (np.sum(e ** 2) + 1e-18) + 1e-18)
            if best is None or snr > best[0]:
                best = (snr, k)
        # pilot on the raw take over the band's span (real time = disc time / speed)
        p0 = play["i0"] + int(b["t0"] / play["speed"] * fs)
        p1 = play["i0"] + int(b["t1"] / play["speed"] * fs)
        pa = amp[p0:p1]
        lev = float(np.median(pa)) if len(pa) else float("nan")
        pulse = float("nan")
        if len(pa) > fs:
            la = np.log(pa / (np.median(pa) + 1e-12) + 1e-6)[:: fs // 50]
            f = np.fft.rfftfreq(len(la) * 4, 1 / 50)
            Y = np.abs(np.fft.rfft((la - la.mean()) * np.hanning(len(la)), len(la) * 4)) * 4 / len(la)
            m = (f > 2.3) & (f < 3.3)
            pulse = float(100 * Y[m].max())
        res.append({"band": b["band"], "run": b["run"], "snr_db": None if best is None else round(best[0], 2),
                    "gain": None if best is None else round(best[1], 4),
                    "pilot_mv": round(1000 * lev, 1), "pulse_2p8hz_pct": round(pulse, 1)})
    return res


def main_effects(man, snr_by_run):
    keys = list(man["factors"])
    grand = float(np.mean(list(snr_by_run.values())))
    eff = {}
    for fi, key in enumerate(keys):
        means = []
        for lev in range(3):
            runs = [r + 1 for r, row in enumerate(man["l9"]) if row[fi] == lev]
            means.append(float(np.mean([snr_by_run[r] for r in runs])))
        best = int(np.argmax(means))
        eff[key] = {"levels": man["factors"][key], "mean_snr_db": [round(m, 2) for m in means],
                    "best_level": man["factors"][key][best], "range_db": round(max(means) - min(means), 2)}
    pred = grand + sum(max(e["mean_snr_db"]) - grand for e in eff.values())
    rank = sorted(keys, key=lambda k: -eff[k]["range_db"])
    return {"grand_mean_snr_db": round(grand, 2), "factors": eff, "rank": rank,
            "best": {k: eff[k]["best_level"] for k in keys}, "predicted_best_snr_db": round(pred, 2)}


def plot(effects, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    keys = list(effects["factors"])
    fig, axs = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 3.0), sharey=True)
    for ax, k in zip(axs, keys):
        e = effects["factors"][k]
        ax.plot(range(3), e["mean_snr_db"], marker="o", color="#2f6f9f", lw=2)
        ax.axhline(effects["grand_mean_snr_db"], color="#999999", lw=1, ls="--")
        ax.set_xticks(range(3), [f"{v:g}" for v in e["levels"]])
        ax.set_title(k.replace("_", " "), fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylabel("mean band SNR, dB")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("takes", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mpath = Path(a.manifest)
    man = json.loads(mpath.read_text())
    out = Path(a.out) if a.out else mpath.parent
    gw = mpath.with_name(mpath.name.replace(".manifest.json", ".groove.wav"))
    rfs, ref = wavfile.read(gw)
    ref = ref.astype(float) / (np.abs(ref).max() + 1e-12)
    per_play = []
    for t in a.takes:
        fs, x = load_take(t)
        for k, play in enumerate(plays_of(x, fs, man)):
            dur = len(play["audio"]) / fs
            if dur < man["audio_s"] * 0.9:
                print(f"{Path(t).name} play {k + 1}: {dur:.1f} s, shorter than the disc ({man['audio_s']:.1f} s): "
                      f"skipped (lifted early or lost the pilot)")
                continue
            m = band_metrics(play, fs, man, ref, rfs)
            per_play.append({"take": Path(t).name, "play": k + 1, "pitch": play["info"], "bands": m})
            print(f"{Path(t).name} play {k + 1}: " + "  ".join(f"b{b['band']}(run {b['run']}) {b['snr_db']:+.1f} dB"
                                                                for b in m))
    if not per_play:
        raise SystemExit("no full play of the disc found")
    snr_by_run = {}
    for b in man["bands"]:
        vals = [pb["snr_db"] for p in per_play for pb in p["bands"] if pb["run"] == b["run"] and pb["snr_db"] is not None]
        snr_by_run[b["run"]] = float(np.mean(vals))
    eff = main_effects(man, snr_by_run)
    rep = {"plays": len(per_play), "snr_by_run_db": {str(k): round(v, 2) for k, v in sorted(snr_by_run.items())},
           "main_effects": eff, "per_play": per_play}
    (out / "taguchi_report.json").write_text(json.dumps(rep, indent=1))
    plot(eff, out / "taguchi_main_effects.png")
    print(f"\n{len(per_play)} play(s). SNR by run:", rep["snr_by_run_db"])
    for k in eff["rank"]:
        e = eff["factors"][k]
        print(f"  {k:12s} levels {e['levels']}  mean SNR {e['mean_snr_db']} dB  best {e['best_level']}  "
              f"range {e['range_db']} dB")
    print(f"best combination {eff['best']}, predicted {eff['predicted_best_snr_db']:+.1f} dB "
          f"(grand mean {eff['grand_mean_snr_db']:+.1f})")
    print(f"wrote {out / 'taguchi_report.json'} and {out / 'taguchi_main_effects.png'}")


if __name__ == "__main__":
    main()
