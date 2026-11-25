"""Music for the printed turntable: conditioning that follows the groove's measured limits, plus a pilot tone the
playback cleanup (restore.py) uses to undo the platter's wow.

What the spine turntable's takes 4 and 5 (Nov 22-24 2026) measured, and what this chain does about each:
  - Tones from 98 to 698 Hz all play; the nozzle-smear estimate (v / bead width, 401 Hz at the label) was too
    pessimistic. What really caps the top end is the 0.1 mm G-code segment (a wavelength needs about four of them) and
    the inner bead's bend radius (A f^2 <= kappa v^2). Both scale with groove speed, so the low-pass here follows the
    radius: about 880 Hz at the outer edge, 480 Hz at the label, instead of 401 Hz everywhere.
  - The bend budget is spent on high notes first. Where it runs short, only the band above 250 Hz is turned down,
    rather than the encoder's limiter ducking the whole signal.
  - The 110 Hz ladder stays linear to about 40 um and compresses beyond (THD 21 % at 40 um, 80 % at 80 um), and the
    surface noise in the music band is about 120 mV against about 5 mV per um. So the music is levelled and
    compressed hard (quiet passages up by as much as 18 dB) and its peaks are 40 um.
  - The bend limit is what the printer can lay down, not what the stylus can follow. Take 6 (Nov 24, 40 um music,
    12 deg slopes, 130-200 m/s^2) bounced out of the groove; the tone disc had tracked about 9 deg and 90 m/s^2. So
    track_limit holds the wall slope to 7 deg and the stylus acceleration to 70 m/s^2 (music peaks fall to about 33 um).
  - An off-centre disc gives a once-per-turn pitch wobble of 0.6 to 1 %. A 75 Hz pilot tone, 10 um, runs under the
    music (below its 120 Hz high-pass and 15 Hz clear of 60 Hz hum). restore.py tracks it and resamples the take.
"""

import math

import numpy as np
from scipy import signal

from .dsp import compress, highpass, robust_normalize

PILOT_HZ = 75.0
SEG_PER_WAVE = 4.0          # G-code segments per wavelength at the low-pass corner
SPLIT_HZ = 250.0            # the band above this gives way first when the bend budget runs short


def play_radius(t, geom):
    """Groove radius (mm) at audio time t (s), after the lead-in turns."""
    theta = (t + geom.lead_in_time) * geom.omega
    return geom.r_of_theta(theta)


def cutoff_hz(r, geom):
    """Low-pass corner at radius r: four segments per wavelength, at most the nozzle-smear figure x 1.2."""
    v = geom.v_play(r)
    return np.minimum(v / (SEG_PER_WAVE * geom.p.segment_length), 1.2 * v / geom.w_e)


def kappa_max(geom):
    """The encoder's bend limit on d2x/ds2 (1/mm): groove.build_groove's curvature limiter."""
    return 0.7 / geom.bead_spacing - 1.0 / geom.p.inner_groove_radius


def level(x, fs, window_s=0.4, max_gain_db=18.0, target_db=-14.0):
    """Slow leveller: lifts quiet passages towards a target RMS (dB re full scale), at most max_gain_db."""
    n = max(1, int(window_s * fs))
    rms = np.sqrt(np.convolve(x * x, np.ones(n) / n, mode="same") + 1e-12)
    g_db = np.clip(target_db - 20 * np.log10(rms + 1e-9), -12.0, max_gain_db)
    g = 10 ** (g_db / 20)
    # smooth the gain so it does not modulate the waveform audibly
    sos = signal.butter(2, 1.5 / (fs / 2), output="sos")
    g = signal.sosfiltfilt(sos, g)
    return x * g


def tracking_lowpass(x, fs, t0_audio, geom, nperseg=2048):
    """STFT mask: a raised-cosine roll-off from 0.8 fc to fc, fc following the radius frame by frame."""
    f, t, X = signal.stft(x, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    r = play_radius(t0_audio + t, geom)
    fc = cutoff_hz(r, geom)
    lo = 0.8 * fc
    F = f[:, None]
    m = np.where(F <= lo[None, :], 1.0,
                 np.where(F >= fc[None, :], 0.0, 0.5 + 0.5 * np.cos(math.pi * (F - lo[None, :]) / (fc - lo)[None, :])))
    _, y = signal.istft(X * m, fs, nperseg=nperseg, noverlap=nperseg * 3 // 4)
    return y[: len(x)], fc


def bend_tilt(x_mm, fs, t0_audio, geom, margin=0.85, win_s=0.02, acc_max=None):
    """Turn the band above SPLIT_HZ down where the groove's bend (d2x/ds2 = x''(t) / v^2) would pass the limit, or,
    with acc_max (mm/s^2, a number or an array), where the stylus's sideways acceleration would.
    x_mm is the groove displacement in mm. Returns (y_mm, fraction of time the tilt acted)."""
    sos = signal.butter(4, SPLIT_HZ / (fs / 2), output="sos")
    lo = signal.sosfiltfilt(sos, x_mm)
    hi = x_mm - lo
    t = t0_audio + np.arange(len(x_mm)) / fs
    v = geom.v_play(play_radius(t, geom))
    allowed = margin * kappa_max(geom) * v * v                       # mm/s^2
    if acc_max is not None:
        allowed = np.minimum(allowed, acc_max)
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    w = max(3, int(win_s * fs))
    acc = lambda y: maximum_filter1d(np.abs(np.gradient(np.gradient(y)) * fs * fs), w)
    a_lo, a_hi = acc(lo), acc(hi)
    g = np.clip((allowed - a_lo) / np.maximum(a_hi, 1e-9), 0.05, 1.0)
    g = uniform_filter1d(maximum_filter1d(-g, w) * -1, 2 * w)        # running min, then smoothed
    return lo + g * hi, float(np.mean(g < 0.99))


def track_limit(x_mm, fs, t0_audio, geom, a_pilot_mm=0.0, slope_deg=7.0, acc_max_ms2=70.0, win_s=0.02, iters=3):
    """Keep the groove inside what the stylus can follow (take 6, Nov 24 2026: the 40 um music groove asked for
    12 deg slopes and 130-200 m/s^2 and the stylus bounced out; the tone disc had tracked about 9 deg and 90 m/s^2).
    First the band above SPLIT_HZ is tilted down where the sideways acceleration would pass acc_max, then a smooth
    gain holds both the wall slope (stylus velocity / groove speed) and the acceleration under their limits.
    The pilot's own share (a_pilot_mm at PILOT_HZ) comes off both budgets. Returns (y_mm, info)."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    t = t0_audio + np.arange(len(x_mm)) / fs
    v = geom.v_play(play_radius(t, geom))                             # mm/s
    wp = 2 * math.pi * PILOT_HZ
    vel_budget = math.tan(math.radians(slope_deg)) * v - wp * a_pilot_mm          # mm/s
    acc_budget = acc_max_ms2 * 1000.0 - wp * wp * a_pilot_mm                       # mm/s^2
    y, tilt = bend_tilt(x_mm, fs, t0_audio, geom, margin=1.0, acc_max=acc_budget)
    w = max(3, int(win_s * fs))
    g_all = np.ones_like(y)
    for _ in range(iters):
        vel = np.gradient(y) * fs
        acc = np.gradient(vel) * fs
        need = np.minimum(vel_budget / (maximum_filter1d(np.abs(vel), w) + 1e-9),
                          acc_budget / (maximum_filter1d(np.abs(acc), w) + 1e-9))
        g = np.minimum(1.0, 0.97 * need)
        g = uniform_filter1d(-maximum_filter1d(-g, 2 * w), 2 * w)       # running minimum, then smoothed
        y = y * g
        g_all *= g
    vel = np.abs(np.gradient(y) * fs) + wp * a_pilot_mm
    acc = np.abs(np.gradient(np.gradient(y) * fs) * fs) + wp * wp * a_pilot_mm
    slope = np.degrees(np.arctan(vel / v))
    info = {"track_slope_deg": slope_deg, "track_acc_ms2": acc_max_ms2, "track_tilt_fraction": round(tilt, 4),
            "track_gain_mean_db": round(float(20 * np.log10(np.mean(g_all))), 1),
            "track_gain_p10_db": round(float(20 * np.log10(np.percentile(g_all, 10))), 1),
            "slope_p99_deg": round(float(np.percentile(slope, 99)), 2), "slope_max_deg": round(float(slope.max()), 2),
            "acc_p99_ms2": round(float(np.percentile(acc, 99)) / 1000, 1), "acc_max_ms2": round(float(acc.max()) / 1000, 1)}
    return y, info


def virtual_bass(x, fs, lo=40.0, hi=120.0, out_lo=130.0, out_hi=360.0, gain=0.6):
    """Bass the groove cannot carry, heard through its harmonics (the missing-fundamental effect, as small speakers
    use). The 40-120 Hz band is rectified (2nd and 4th harmonics) and cubed against its own running level (3rd), both
    of degree one in amplitude so loud and quiet bass keep their balance, then band-passed to 130-360 Hz: above the
    75 Hz pilot and inside the groove's best band. gain sets their RMS against the bass they stand for.
    Returns (harmonics to add, bass RMS)."""
    sos = signal.butter(4, [lo / (fs / 2), hi / (fs / 2)], "band", output="sos")
    b = signal.sosfiltfilt(sos, x)
    n = max(1, int(0.05 * fs))
    rms = np.sqrt(np.convolve(b * b, np.ones(n) / n, mode="same") + 1e-12)
    h = np.abs(b) + 0.5 * b ** 3 / (rms * rms)
    sos2 = signal.butter(4, [out_lo / (fs / 2), out_hi / (fs / 2)], "band", output="sos")
    h = signal.sosfiltfilt(sos2, h)
    return gain * h * (b.std() / (h.std() + 1e-12)), float(b.std())


def peak_limit(x, fs, ceiling=0.5, look_ms=5.0, release_ms=40.0):
    """Look-ahead peak limiter: a smooth gain (no clipping, so no new harmonics to spend the bend budget on) that
    holds peaks to ceiling x the current peak, so after normalising the RMS rises by up to 1/ceiling."""
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    p = np.abs(x).max()
    if p <= 0:
        return x
    w = max(1, int(look_ms * 1e-3 * fs))
    env = maximum_filter1d(np.abs(x), 2 * w + 1)
    g = np.minimum(1.0, ceiling * p / (env + 1e-12))
    g = -maximum_filter1d(-g, 2 * w + 1)                     # hold the minimum across the look-ahead window
    a = 1.0 - math.exp(-1.0 / (release_ms * 1e-3 * fs))
    out = np.empty_like(g)                                   # instant attack (already ahead), smooth release
    out[0] = g[0]
    for i in range(1, len(g)):
        out[i] = g[i] if g[i] < out[i - 1] else out[i - 1] + a * (g[i] - out[i - 1])
    out = uniform_filter1d(out, w)
    return x * out


def prepare(audio, fs, geom, a_music_mm=0.040, a_pilot_mm=0.010, hpf=120.0, fade_s=0.3, ratio=4.0, bass=0.6, limit=0.5,
            slope_deg=7.0, acc_max_ms2=70.0, t0_audio=0.0, pilot=True):
    """Condition a mono clip for the groove. Returns (signal normalised to +-1, info dict). The groove amplitude to
    cut it at is info["amplitude_mm"]: the tracking limit (slope_deg, acc_max_ms2; slope_deg=None turns it off) leaves
    the peak below a_music + a_pilot. bass: virtual_bass gain (0 turns it off). t0_audio: where the clip starts on
    the disc (audio time, s), for the radius-dependent limits. pilot=False leaves the pilot out (its share of the
    tracking budget is still kept), for a caller that lays one pilot under several clips (the Taguchi disc).
    y * info["amplitude_mm"] is the groove in mm either way."""
    x = audio - np.mean(audio)
    vb, b_rms = virtual_bass(x, fs, gain=bass) if bass > 0 else (0.0, 0.0)
    x = highpass(x, fs, hpf, order=4)
    x_rms = float(np.std(x))
    x = x + vb
    x = robust_normalize(x, fs)
    x = level(x, fs)
    x = compress(x, fs, ratio=ratio, target_dr_db=12.0, attack_ms=3.0, release_ms=80.0)
    x, fc = tracking_lowpass(x, fs, t0_audio, geom)
    x = robust_normalize(x, fs)
    # fades, so the groove does not start on a step
    n = len(x)
    tt = np.arange(n) / fs
    env = np.clip(tt / fade_s, 0, 1) * np.clip((tt[-1] - tt) / fade_s, 0, 1)
    x = x * env
    x_mm, frac = bend_tilt(x * a_music_mm, fs, t0_audio, geom)
    x = x_mm / a_music_mm
    crest_in = float(np.abs(x).max() / (x.std() + 1e-12))
    x = peak_limit(x, fs, ceiling=limit)
    p = np.abs(x).max()
    x = x / p if p > 0 else x
    m_mm = a_music_mm * x
    tinfo = {}
    if slope_deg is not None:
        m_mm, tinfo = track_limit(m_mm, fs, t0_audio, geom, a_pilot_mm, slope_deg, acc_max_ms2)
    pilot_w = np.sin(2 * math.pi * PILOT_HZ * tt) * np.clip(tt / 0.05, 0, 1) * np.clip((tt[-1] - tt) / 0.05, 0, 1)
    y_mm = m_mm + a_pilot_mm * pilot_w if pilot else m_mm
    a_tot = float(np.abs(y_mm).max())
    y = y_mm / a_tot
    info = {"pilot_hz": PILOT_HZ, "a_music_um": round(float(np.abs(m_mm).max()) * 1000, 1),
            "a_music_rms_um": round(float(m_mm.std()) * 1000, 1), "a_pilot_um": a_pilot_mm * 1000,
            "amplitude_mm": round(a_tot, 5), **tinfo, "hpf_hz": hpf, "lpf_outer_hz": round(float(fc[0]), 1),
            "lpf_inner_hz": round(float(fc[-1]), 1), "bend_tilt_fraction": round(frac, 4),
            "audio_s": round(n / fs, 3), "fs": fs, "virtual_bass_gain": bass,
            "crest_before_limit": round(crest_in, 2), "crest": round(float(np.abs(x).max() / (x.std() + 1e-12)), 2),
            "bass_below_hpf_vs_kept_db": round(20 * math.log10((b_rms + 1e-12) / (x_rms + 1e-12)), 1)}
    return y, info
