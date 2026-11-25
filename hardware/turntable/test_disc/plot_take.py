"""
plot_take.py - spectrogram of a test-disc take with the disc's sections marked where they should play.

    python plot_take.py TAKE.wav MANIFEST.json T0 OUT.png [--end 33] [--rpm 39.84]

T0 is where the disc's play time 0 falls in the take (seconds), as found by lining up the 198 Hz tone. Used for
the portfolio's first-play figure (take 4, Nov 22 2026: T0 = -0.54 on the printed disc's manifest).
"""
import argparse
import json
import wave

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import spectrogram


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("take")
    ap.add_argument("manifest")
    ap.add_argument("t0", type=float)
    ap.add_argument("out")
    ap.add_argument("--end", type=float, default=33.0)
    ap.add_argument("--rpm", type=float, default=39.84)
    a = ap.parse_args()
    with wave.open(a.take) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767 * 2.5
        fs = w.getframerate()
    x = x[: int(a.end * fs)] - x.mean()
    f, t, S = spectrogram(x, fs, nperseg=2048, noverlap=1792, window="hann")
    keep = (f >= 60) & (f <= 900)
    db = 10 * np.log10(S[keep] + 1e-12)
    m = json.load(open(a.manifest))
    k = a.rpm / m["rpm"]

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(8.0, 4.4), dpi=160)
    ax.pcolormesh(t, f[keep], db, shading="auto", cmap="Greys", vmin=np.percentile(db, 40), vmax=np.percentile(db, 99.7))
    groups = [("reference outer", "333 Hz"), ("tone 698 Hz", "698"), ("tone 627 Hz", ""), ("tone 498 Hz", ""), ("tone 398 Hz", ""),
              ("tone 333 Hz", ""), ("tone 198 Hz", "198"), ("tone 147 Hz", "147"), ("tone 98 Hz", "98"),
              ("ladder 10 um", ""), ("ladder 20 um", ""), ("ladder 40 um", ""), ("ladder 60 um", ""),
              ("ladder 80 um", ""), ("reference inner", "333 Hz")]
    S_ = {s["kind"]: s for s in m["sections"]}
    for kind, label in groups:
        s = S_.get(kind)
        if not s:
            continue
        t0, t1 = a.t0 + s["t0"] / k, a.t0 + s["t1"] / k
        if t1 > a.end:
            continue
        fr = s["f_hz"] * k
        ax.plot([t0, t1], [fr, fr], color="#c0392b", lw=1.4, alpha=0.75, ls=(0, (2, 2)))
        if label:
            ax.text((t0 + t1) / 2, fr + 14, label, ha="center", va="bottom", fontsize=10, color="#c0392b")
    lad = [S_[f"ladder {u} um"] for u in (10, 80) if f"ladder {u} um" in S_]
    if len(lad) == 2:
        ax.text(a.t0 + (lad[0]["t0"] + lad[1]["t1"]) / 2 / k, 110 * k + 16, "110 Hz, 10 to 80 µm", ha="center", va="bottom", fontsize=10, color="#c0392b")
    ax.set_xlim(0, a.end)
    ax.set_ylim(60, 900)
    ax.set_xlabel("time in the take (s)")
    ax.set_ylabel("frequency (Hz)")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(a.out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
