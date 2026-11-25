"""Playback cleanup for a take of a song disc (song.py), recorded by the Arduino (adc_capture/capture.py).

    python -m openvinyl.restore TAKE.wav [TAKE2.wav ...] MANIFEST.json -o OUT.wav

Play the disc several times (several needle drops in one take, or several takes): every pass is cleaned and they
are averaged, which lowers the surface noise by about 3 dB per doubling (it differs from play to play).

Steps, in the order the faults arise:
  1. hum and buzz: 2 Hz notches on every 60 Hz harmonic (mains, in recording time, so before the resample);
  2. clicks: the music has nothing above about 900 Hz, so a click shows plainly in the band above 1.2 kHz. Each one
     is cut out (a few ms) and filled by least-squares AR interpolation from the samples either side;
  3. wow: the 75 Hz pilot's instantaneous frequency gives the playback speed. It is fitted with the off-centre model
     (once and twice per turn, amplitude growing as 1/r towards the label) plus a slow drift, and the take is
     resampled onto the disc's own clock. This also removes the platter's steady speed error;
  4. band: the pilot and everything outside the music's band (the same radius-following low-pass the encoder used)
     is filtered out;
  5. passes: every needle drop is cleaned and the passes are averaged (the noise differs between plays);
  6. noise: a Wiener gain from the noise in the lead-in turn plus the part of the take's power that repeats with the
     turn (the infill crossings under the groove, twice per turn), so the whoosh that pulses with the rotation is
     pressed down where it happens.
"""

import json
import math

import numpy as np
from scipy import signal
from scipy.ndimage import maximum_filter1d, median_filter, uniform_filter1d


def notch_hum(x, fs, mains=60.0, bw=2.0):
    """Mains hum and buzz. The Arduino picks up spikes every 16.7 ms (odd harmonics of 60 Hz all the way up), so
    every harmonic to the Nyquist rate gets a notch of constant width bw (Hz): 2 Hz of each 60, about 3 % of the
    band, and narrow enough for the mains' own drift."""
    for k in range(1, int((fs / 2 - bw) // mains) + 1):
        f = k * mains
        b, a = signal.iirnotch(f, f / bw, fs)
        x = signal.filtfilt(b, a, x)
    return x


def _ar_fill(x, i0, i1, p=24, ctx=None):
    """Fill x[i0:i1] by least-squares AR interpolation (Janssen et al.): fit AR(p) on the context either side, then
    choose the missing samples to minimise the total prediction error."""
    m = i1 - i0
    ctx = ctx or max(8 * p, 4 * m)
    a0, b1 = max(0, i0 - ctx), min(len(x), i1 + ctx)
    seg = x[a0:b1].copy()
    known = np.ones(len(seg), bool)
    known[i0 - a0:i1 - a0] = False
    # AR fit on the known samples (covariance method on each side)
    rows, rhs = [], []
    for j in range(p, len(seg)):
        if known[j - p:j + 1].all():
            rows.append(seg[j - p:j][::-1])
            rhs.append(seg[j])
    if len(rows) < 2 * p:
        seg[~known] = np.interp(np.nonzero(~known)[0], np.nonzero(known)[0], seg[known])
        x[a0:b1] = seg
        return
    a, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    # prediction-error filter e = A s, with s split into known and unknown parts
    n = len(seg)
    coef = np.concatenate([[1.0], -a])
    A = np.zeros((n - p, n))
    for k in range(n - p):
        A[k, k:k + p + 1] = coef[::-1]
    Au, Ak = A[:, ~known], A[:, known]
    su, *_ = np.linalg.lstsq(Au, -Ak @ seg[known], rcond=None)
    lim = 1.5 * np.abs(seg[known]).max()
    if not np.all(np.isfinite(su)) or np.abs(su).max() > lim:       # an unstable fit: fall back to a straight line
        su = np.interp(np.nonzero(~known)[0], np.nonzero(known)[0], seg[known])
    seg[~known] = su
    x[a0:b1] = seg


def declick(x, fs, hp_hz=1200.0, ratio=4.0, pad_ms=1.0, max_ms=5.0, abs_v=0.02):
    """Find clicks in the band above hp_hz and fill them. A click is a short burst: its envelope stands ratio times
    above the median of the surrounding 50 ms and above abs_v (volts; 4 of the UNO's 4.9 mV steps), and it lasts
    under max_ms (sustained high-band content, such as a
    note's harmonics or the pickup's distortion, raises the median with it and is left alone).
    Returns (x, number of clicks, fraction of samples filled)."""
    sos = signal.butter(4, hp_hz / (fs / 2), "high", output="sos")
    h = np.abs(signal.sosfiltfilt(sos, x))
    env = maximum_filter1d(h, max(1, int(0.0005 * fs)))
    med = median_filter(env, size=int(0.05 * fs), mode="nearest")
    floor = np.median(env)
    bad = env > np.maximum(ratio * np.maximum(med, floor), abs_v)
    pad = int(pad_ms * 1e-3 * fs)
    bad = maximum_filter1d(bad.astype(np.uint8), 2 * pad + 1).astype(bool)
    y = x.copy()
    edges = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
    starts, ends = np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]
    nmax = int(max_ms * 1e-3 * fs)
    keep = (ends - starts) <= nmax
    for s, e in zip(starts[keep], ends[keep]):
        _ar_fill(y, s, e)
    filled = int(np.sum((ends - starts)[keep]))
    return y, int(keep.sum()), filled / len(x)


def pilot_phase(x, fs, f0, bw=3.0):
    """The pilot brought to baseband: z = lowpass_bw(x e^{-j 2 pi f0 t}). Returns (unwrapped phase in cycles,
    amplitude). A 3 Hz band holds the once- and twice-per-turn wobble (0.66 and 1.33 Hz) and keeps the noise power
    five times below a +-8 Hz band-pass, so the phase rarely slips a cycle."""
    t = np.arange(len(x)) / fs
    sos = signal.butter(4, bw / (fs / 2), output="sos")
    zr = signal.sosfiltfilt(sos, x * np.cos(2 * math.pi * f0 * t))
    zi = signal.sosfiltfilt(sos, -x * np.sin(2 * math.pi * f0 * t))
    z = zr + 1j * zi
    return np.unwrap(np.angle(z)) / (2 * math.pi), 2 * np.abs(z)


def fit_speed(t, cyc, w, f0, rpm, r_of_t, knots_s=4.0):
    """Robust least-squares fit of the pilot's phase (cycles), as the playback clock's lead over real time:
        cyc(t) / f0 = tau(t) - t = drift(t) + (r0 / r(t)) * sum_{k=1,2} (a_k sin(k W t) - b_k cos(k W t)) / (k W)
    so the speed is d tau / dt = 1 + drift'(t) + (r0 / r) sum a_k cos(k W t) + b_k sin(k W t): an off-centre disc
    (once and twice per turn, growing as 1/r towards the label) and a slow drift, a cubic B-spline with knots every
    knots_s seconds. w are per-sample weights (pilot present). Returns (speed at t, info)."""
    from scipy.interpolate import BSpline
    d = cyc / f0
    W = 2 * math.pi * rpm / 60.0
    s = r_of_t(t[0]) / r_of_t(t)
    cols, dcols = [], []
    for k in (1, 2):
        cols += [s * np.sin(k * W * t) / (k * W), -s * np.cos(k * W * t) / (k * W)]
        dcols += [s * np.cos(k * W * t), s * np.sin(k * W * t)]
    n_k = max(2, int(math.ceil((t[-1] - t[0]) / knots_s)))
    inner = np.linspace(t[0], t[-1], n_k + 1)
    kn = np.concatenate([[t[0]] * 3, inner, [t[-1]] * 3])
    for i in range(len(kn) - 4):
        c = np.zeros(len(kn) - 4)
        c[i] = 1
        bs = BSpline(kn, c, 3, extrapolate=True)
        cols.append(bs(t))
        dcols.append(bs.derivative()(t))
    M, dM = np.column_stack(cols), np.column_stack(dcols)
    wt = w.copy()
    for _ in range(8):                                  # iteratively reweighted (Huber): phase slips and dropouts
        coef, *_ = np.linalg.lstsq(M * wt[:, None], d * wt, rcond=None)
        res = d - M @ coef
        sc = 1.4826 * np.median(np.abs(res[w > 0] - np.median(res[w > 0]))) + 1e-12
        wt = w * np.minimum(1.0, 1.5 * sc / (np.abs(res) + 1e-15))
    speed = 1.0 + dM @ coef
    return speed, {"wow_once_pct": round(100 * float(np.hypot(coef[0], coef[1])), 3),
                   "wow_twice_pct": round(100 * float(np.hypot(coef[2], coef[3])), 3),
                   "phase_residual_ms": round(1000 * float(np.std(res[w > 0])), 3)}


def warp(x, fs, speed, up=4):
    """Resample x so that time runs at the disc's clock: disc time tau(t) = integral of speed dt."""
    xu = signal.resample_poly(x, up, 1)
    t = np.arange(len(xu)) / (fs * up)
    sp = np.interp(t, np.arange(len(speed)) / fs, speed)
    tau = np.concatenate([[0], np.cumsum(0.5 * (sp[1:] + sp[:-1]) / (fs * up))])
    tau_out = np.arange(0, tau[-1], 1.0 / fs)
    return np.interp(tau_out, tau, xu)


def wiener(x, fs, noise, alpha=1.5, floor_db=-12.0, nperseg=512, noise_scale=1.0):
    """Mild spectral noise reduction from a noise example (same sample rate). noise_scale multiplies the noise
    power: 1/N after averaging N plays whose noise is independent."""
    _, _, N = signal.stft(noise, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    pn = noise_scale * np.mean(np.abs(N) ** 2, axis=1, keepdims=True)
    f, t, X = signal.stft(x, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    px = uniform_filter1d(np.abs(X) ** 2, 3, axis=1)
    g = np.maximum(1 - alpha * pn / (px + 1e-12), 10 ** (floor_db / 20))
    g = uniform_filter1d(g, 3, axis=1)
    _, y = signal.istft(X * g, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    return y[: len(x)]


def wiener_sync(x, fs, noise, rpm, alpha=1.5, floor_db=-15.0, nperseg=512, noise_scale=1.0, harmonics=4):
    """Spectral noise reduction for noise whose loudness follows the platter. In takes 4 and 5 the music-band noise
    power swings 10-14x over a turn (the straight infill lines under the groove, crossed twice per turn; one crossing
    louder), which plays as a whoosh pulsing with the rotation. x must be on the disc's clock (after warp), so turn
    phase is t * rpm / 60. Power that is periodic with the turn belongs to the noise (the music is not locked to the
    platter): per frequency bin, |X|^2 is fitted over the pass with turn harmonics 1..harmonics, and the noise
    estimate is the lead-in's mean noise plus that periodic part, so the gain clamps down on each crossing."""
    _, _, N = signal.stft(noise, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    n0 = noise_scale * np.mean(np.abs(N) ** 2, axis=1)
    f, t, X = signal.stft(x, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    P = np.abs(X) ** 2
    ph = 2 * math.pi * t * rpm / 60.0
    B = [np.ones_like(t)]
    for k in range(1, harmonics + 1):
        B += [np.cos(k * ph), np.sin(k * ph)]
    B = np.column_stack(B)
    C, *_ = np.linalg.lstsq(B, P.T, rcond=None)              # (2K+1) x F
    periodic = (B[:, 1:] @ C[1:]).T                          # F x T
    pn = np.maximum(n0[:, None] + periodic, 0.05 * n0[:, None])
    px = uniform_filter1d(P, 3, axis=1)
    g = np.maximum(1 - alpha * pn / (px + 1e-12), 10 ** (floor_db / 20))
    g = uniform_filter1d(g, 3, axis=1)
    _, y = signal.istft(X * g, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    swing = float(np.median((n0[:, None] + periodic).max(axis=1) / np.maximum((n0[:, None] + periodic).min(axis=1), 1e-3 * n0)))
    return y[: len(x)], swing


def find_passes(x, fs, f0, min_s=8.0):
    """Stretches where the pilot plays (one per needle drop), as [(i0, i1)], and the pilot's phase and amplitude."""
    cyc, amp = pilot_phase(x, fs, f0)
    # smoothed over most of a turn (the pilot's level swings with the turn), and gaps under 2 s are dips, not lifts
    # (take 8, Nov 24: a tracking pilot of 25-40 mV, quieter than the lift spikes, split into pieces at 0.5 s)
    a_s = uniform_filter1d(amp, int(1.2 * fs))
    on = a_s > 0.4 * np.percentile(a_s, 90)
    edges = np.diff(np.concatenate([[0], on.astype(int), [0]]))
    merged = []
    for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
        if merged and a - merged[-1][1] < 2.0 * fs:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [(int(a), int(b)) for a, b in merged if b - a >= min_s * fs], cyc, amp


def _cutoff_at(tt, man, r_of_t):
    from .song import cutoff_hz

    class G:                                           # what cutoff_hz needs
        pass
    g = G()
    g.p = G()
    g.p.segment_length = man.get("segment_mm", 0.1)
    g.w_e = man.get("bead_width_mm", 0.48)
    g.v_play = lambda r: 2 * math.pi * man["rpm"] / 60.0 * r
    return cutoff_hz(r_of_t(tt), g)


def restore_pass(x, fs, man, i0, i1, cyc, amp, steps):
    """One pass onto the disc's clock and into the music's band. Returns (audio, noise example, info)."""
    info = {"pass_s": [round(i0 / fs, 2), round(i1 / fs, 2)]}
    f0, rpm = man["pilot_hz"], man["rpm"]
    lead = int(60.0 / rpm * fs)                            # one lead-in turn before the pilot: the noise example
    noise = x[max(0, i0 - lead - int(0.15 * fs)):max(0, i0 - int(0.15 * fs))]
    x = x[i0:i1]
    cyc, amp = cyc[i0:i1], amp[i0:i1]
    t = np.arange(len(x)) / fs
    pitch, r_out = man["pitch_mm"], man["r_outer_mm"]
    lead_turns = man.get("lead_in_turns", 1.0)
    r_of_t = lambda tt: r_out - pitch * (lead_turns + tt * rpm / 60.0)
    if "wow" in steps:
        w = (amp > 0.5 * np.median(amp)).astype(float)
        dec = max(1, int(fs / 100))                       # fit at 100 Hz: the pilot's band is 3 Hz
        tt, cc, ww = t[::dec], cyc[::dec], w[::dec]
        slope = np.polyfit(tt[ww > 0], cc[ww > 0], 1)[0]  # cycles/s: the mean speed error times f0
        rpm_true = rpm * (1 + slope / f0)                 # the platter's own speed sets the once-per-turn rate
        sp, wi = fit_speed(tt, cc, ww, f0, rpm_true, r_of_t)
        speed = np.interp(t, tt, sp)
        if wi["phase_residual_ms"] > 1.5:
            # the off-centre model does not hold: something else wobbles the pitch (take 8, Nov 24: +-2.8 Hz sidebands
            # on the pilot, the arm rocking under the added weight). Follow the pilot itself: a wider band holds the
            # sidebands (the 3 Hz one slips cycles on them), and the clock is its phase smoothed to 8 Hz.
            cw, aw = pilot_phase(x, fs, f0, bw=7.0)
            good = aw > 0.4 * np.median(aw)
            cw = np.interp(t, t[good], cw[good])               # bridge dropouts on the line between good stretches
            sos = signal.butter(2, 8.0 / (fs / 2), output="sos")
            speed = 1.0 + signal.sosfiltfilt(sos, np.gradient(cw) * fs) / f0
            res = (cw - np.polyval(np.polyfit(t, cw, 1), t)) / f0
            wi = {"wow_mode": "direct", "model_residual_ms": wi["phase_residual_ms"],
                  "pitch_wobble_ms_rms": round(1000 * float(np.std(res)), 3)}
        info.update(wi, speed_error_pct=round(100 * (float(np.mean(speed)) - 1), 3))
        x = warp(x, fs, speed)
        if len(noise) > 0:
            noise = warp(noise, fs, np.full(len(noise), float(np.mean(speed))))
    t = np.arange(len(x)) / fs
    if "band" in steps:
        sos = signal.butter(6, (man.get("hpf_hz", 120.0) - 10) / (fs / 2), "high", output="sos")
        x = signal.sosfiltfilt(sos, x)
        noise = signal.sosfiltfilt(sos, noise) if len(noise) > 3 * 50 else noise
        nps = 1024
        ff, tt, X = signal.stft(x, fs, nperseg=nps, noverlap=nps * 3 // 4)
        fc = _cutoff_at(tt, man, r_of_t) * 1.05
        m = uniform_filter1d((ff[:, None] <= fc[None, :]).astype(float), 5, axis=0)
        _, x = signal.istft(X * m, fs, nperseg=nps, noverlap=nps * 3 // 4)
        x = x[: len(t)]
        if len(noise) > 50:
            noise = signal.sosfiltfilt(signal.butter(6, float(fc[0]) / (fs / 2), output="sos"), noise)
    return x, noise, info


def _frac_lag(ref, y, max_lag):
    """Lag (samples, fractional) that best lines y up with ref: y[n + lag] ~ ref[n]."""
    c = signal.correlate(y, ref, mode="full", method="fft")
    mid = len(ref) - 1
    lo, hi = mid - max_lag, mid + max_lag + 1
    k = lo + int(np.argmax(c[lo:hi]))
    if lo < k < hi - 1:
        y0, y1, y2 = c[k - 1], c[k], c[k + 1]
        d = 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2 + 1e-18)
    else:
        d = 0.0
    return k - mid + d, float(c[k] / (np.linalg.norm(ref) * np.linalg.norm(y[:len(ref)]) + 1e-18))


def align_average(passes, fs, block_s=1.0, coarse_s=1.5, fine_ms=4.0):
    """Average plays of the same groove. Each pass is first lined up as a whole with the longest one (pilot onsets
    only agree to about 50 ms), then block by block to a fraction of a sample, so leftover wobble does not smear the
    top of the band. Returns (average, number of passes averaged at each sample)."""
    ref = max(passes, key=len)
    n = len(ref)
    acc, cnt = ref.copy(), np.ones(n)
    for y in passes:
        if y is ref:
            continue
        lag0, _ = _frac_lag(ref[: min(n, len(y))], np.concatenate([y, np.zeros(n)]), int(coarse_s * fs))
        lag0 = int(round(lag0))
        blk, hop = int(block_s * fs), int(block_s * fs / 2)
        centres, lags = [], []
        for b0 in range(0, n - blk, hop):
            s0 = b0 + lag0
            seg = y[max(0, s0 - int(fine_ms * 1e-3 * fs)):s0 + blk + int(fine_ms * 1e-3 * fs)]
            if s0 - int(fine_ms * 1e-3 * fs) < 0 or len(seg) < blk + 2 * int(fine_ms * 1e-3 * fs):
                continue
            L, r = _frac_lag(ref[b0:b0 + blk], seg, int(fine_ms * 1e-3 * fs))
            centres.append(b0 + blk / 2)
            lags.append(lag0 - int(fine_ms * 1e-3 * fs) + L)
        if len(lags) < 2:
            continue
        lags = median_filter(np.array(lags), size=3, mode="nearest")
        idx = np.arange(n)
        src = idx + np.interp(idx, centres, lags)
        ok = (src >= 0) & (src <= len(y) - 1)
        acc[ok] += np.interp(src[ok], np.arange(len(y)), y)
        cnt[ok] += 1
    return acc / cnt, cnt


def restore(takes, fs, man, verbose=True, steps=("hum", "click", "wow", "band", "noise"), passes="all"):
    """Clean a song disc's take(s). takes: one array or a list (several recordings of the same disc). Every pass
    (needle drop) of at least 8 s in every take is put on the disc's clock and, when there are several, they are
    averaged: the surface noise differs from play to play (take 5's two passes, lined up, correlate at 0.07 above
    1.5 kHz), so N plays lower it by about 10 log10 N dB. passes="longest" uses one pass only.
    Returns (audio at fs, info)."""
    if isinstance(takes, np.ndarray):
        takes = [takes]
    f0 = man["pilot_hz"]
    outs, noises, infos = [], [], []
    for take in takes:
        x = take - np.mean(take)
        tinfo = {}
        if "hum" in steps:
            x = notch_hum(x, fs)
        if "click" in steps:
            x, n, frac = declick(x, fs)
            tinfo.update(clicks=n, click_fill_pct=round(100 * frac, 2))
        found, cyc, amp = find_passes(x, fs, f0)
        if passes == "longest" and found:
            found = [max(found, key=lambda p: p[1] - p[0])]
        for i0, i1 in found:
            y, nz, pinfo = restore_pass(x, fs, man, i0, i1, cyc, amp, steps)
            outs.append(y)
            noises.append(nz)
            infos.append({**tinfo, **pinfo})
    if not outs:
        raise SystemExit("no pass with the pilot found")
    if len(outs) > 1:
        x, cnt = align_average(outs, fs)
    else:
        x, cnt = outs[0], np.ones(len(outs[0]))
    nz = max(noises, key=len)
    info = {"passes": len(outs)}
    if "noise" in steps and len(nz) > 512:
        x, swing = wiener_sync(x, fs, nz, man["rpm"], noise_scale=1.0 / float(np.median(cnt)))
        info["noise_swing_per_turn"] = round(swing, 1)
    x = x / (np.abs(x).max() + 1e-12) * 0.89
    info["per_pass"] = infos
    if verbose:
        print(f"restore: {len(outs)} pass(es) averaged")
        for i in infos:
            print("  ", i)
    return x, info


def main(argv=None):
    import argparse
    import wave
    from .dsp import write_wav
    ap = argparse.ArgumentParser(prog="openvinyl restore")
    ap.add_argument("takes", nargs="+", help="one or more takes of the same disc, then the manifest")
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("--rate", type=int, default=22050, help="output sample rate")
    ap.add_argument("--skip", default="", help="comma list of steps to skip: hum,click,wow,band,noise")
    a = ap.parse_args(argv)
    *takes, manifest = a.takes
    xs = []
    for path in takes:
        with wave.open(path) as w:
            fs = w.getframerate()
            xs.append(np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(float) / 32767 * 2.5)   # volts
    man = json.load(open(manifest))
    steps = tuple(s for s in ("hum", "click", "wow", "band", "noise") if s not in a.skip.split(","))
    y, info = restore(xs, fs, man, steps=steps)
    from fractions import Fraction
    fr = Fraction(a.rate, fs).limit_denominator(2000)
    y = signal.resample_poly(y, fr.numerator, fr.denominator)
    write_wav(a.output, y, a.rate)
    print("wrote", a.output)


if __name__ == "__main__":
    main()
