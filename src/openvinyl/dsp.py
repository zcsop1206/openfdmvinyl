"""Audio conditioning for lateral-cut groove encoding.

The chain is the v1 chain minus quantization and dither:
  mono mix -> HPF -> LPF -> compressor -> normalize.
Slope limiting happens later in groove.py because it depends on radius.
"""

import wave
import numpy as np
from scipy import signal


def read_wav(path: str):
    """Return (mono float64 samples in [-1, 1], sample_rate)."""
    with wave.open(path, "rb") as wf:
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        fs = wf.getframerate()
        raw = wf.readframes(wf.getnframes())
    if sw == 1:
        s = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    elif sw == 2:
        s = np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0
    elif sw == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v = np.where(v & 0x800000, v - (1 << 24), v)
        s = v.astype(np.float64) / 8388608.0
    elif sw == 4:
        s = np.frombuffer(raw, dtype=np.int32).astype(np.float64) / 2147483648.0
    else:
        raise ValueError(f"unsupported sample width {sw}")
    if ch > 1:
        s = s.reshape(-1, ch).mean(axis=1)
    return s, fs


def write_wav(path: str, samples: np.ndarray, fs: int):
    x = np.clip(samples, -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(fs))
        wf.writeframes(pcm.tobytes())


def _sos(kind: str, cutoff: float, fs: float, order: int = 4):
    nyq = fs / 2.0
    c = min(cutoff, 0.99 * nyq)
    return signal.butter(order, c / nyq, btype=kind, output="sos")


def highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    if cutoff <= 0:
        return x
    return signal.sosfiltfilt(_sos("high", cutoff, fs, order), x)


def lowpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    if cutoff <= 0 or cutoff >= fs / 2:
        return x
    return signal.sosfiltfilt(_sos("low", cutoff, fs, order), x)


def robust_normalize(y: np.ndarray, fs: float, edge_s: float = 0.05) -> np.ndarray:
    """Peak-normalize ignoring filter edge transients, then clip."""
    n = int(edge_s * fs)
    core = y[n:-n] if len(y) > 3 * n else y
    p = np.abs(core).max()
    return np.clip(y / p, -1.0, 1.0) if p > 0 else y


def compress(x: np.ndarray, fs: float, ratio: float = 6.0, target_dr_db: float = 18.0,
             attack_ms: float = 2.0, release_ms: float = 50.0,
             control_rate: float = 2000.0) -> np.ndarray:
    """Feed-forward RMS compressor. Gain is computed at a low control rate
    (attack/release loop) and interpolated back, so long files stay fast."""
    if len(x) == 0:
        return x
    dec = max(1, int(fs // control_rate))
    n_ctrl = int(np.ceil(len(x) / dec))
    sq = x ** 2
    pad = n_ctrl * dec - len(x)
    rms = np.sqrt(np.pad(sq, (0, pad)).reshape(n_ctrl, dec).mean(axis=1) + 1e-12)
    peak = rms.max()
    if peak <= 0:
        return x
    rms_db = 20.0 * np.log10(rms / peak + 1e-12)
    thr = -target_dr_db
    gain_db = np.where(rms_db > thr, (rms_db - thr) * (1.0 - 1.0 / ratio), 0.0)
    target = 10.0 ** (-gain_db / 20.0)

    fc = fs / dec
    a_att = 1.0 - np.exp(-1000.0 / (attack_ms * fc))
    a_rel = 1.0 - np.exp(-1000.0 / (release_ms * fc))
    g = np.empty_like(target)
    g[0] = target[0]
    for i in range(1, len(target)):
        a = a_att if target[i] < g[i - 1] else a_rel
        g[i] = g[i - 1] + a * (target[i] - g[i - 1])
    t_ctrl = np.arange(n_ctrl) * dec
    gain = np.interp(np.arange(len(x)), t_ctrl, g)
    return robust_normalize(x * gain, fs)


def condition(x: np.ndarray, fs: float, hpf: float = 300.0, lpf: float = 680.0,
              ratio: float = 6.0, target_dr_db: float = 18.0) -> np.ndarray:
    """Full conditioning chain. Returns peak-normalized mono in [-1, 1]."""
    y = x - np.mean(x)
    y = highpass(y, fs, hpf)
    y = lowpass(y, fs, lpf)
    if ratio > 1.0:
        y = compress(y, fs, ratio=ratio, target_dr_db=target_dr_db)
    return robust_normalize(y, fs)


def test_signal(kind: str, duration: float, fs: int = 44100) -> np.ndarray:
    """Synthetic test patterns for characterization discs.

    kind: 'silence' | 'tone:<hz>' | 'sweep:<f0>:<f1>' | 'ladder:<f0>:<f1>:<n>'
    """
    t = np.arange(int(duration * fs)) / fs
    if kind == "silence":
        return np.zeros_like(t)
    parts = kind.split(":")
    if parts[0] == "tone":
        return np.sin(2 * np.pi * float(parts[1]) * t)
    if parts[0] == "sweep":
        f0, f1 = float(parts[1]), float(parts[2])
        return signal.chirp(t, f0, duration, f1, method="logarithmic")
    if parts[0] == "ladder":
        f0, f1, n = float(parts[1]), float(parts[2]), int(parts[3])
        freqs = np.geomspace(f0, f1, n)
        seg = len(t) // n
        out = np.zeros_like(t)
        for i, f in enumerate(freqs):
            sl = slice(i * seg, (i + 1) * seg)
            out[sl] = np.sin(2 * np.pi * f * t[sl])
        return out
    raise ValueError(f"unknown test signal '{kind}'")
