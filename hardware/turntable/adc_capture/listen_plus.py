"""
listen_plus.py - listen.py with a closer-to-the-record live monitor. Records the same files; listen.py is unchanged.

    python listen_plus.py COM15 50 take34.wav     # needs: pip install sounddevice
    python listen_plus.py --sim take30.wav out.wav   the same processing over a saved take

The live chain, causal, block by block (Nov 25 2026), about 75 ms later than listen.py:
  120 Hz high-pass, 75 Hz and mains notches   as listen.py
  click filling    bursts above 1.2 kHz, up to 5 ms, refilled from the samples either side (20 ms look-ahead)
  equaliser        a fixed gain per frequency in place of the 1 kHz low-pass, 150-1000 Hz, fitted against the original
                   soundtrack of the libets_delay disc (takes 18, 24, 30, 33): it suits THIS song and pickup
  noise reduction  only between songs, up to 18 dB; the noise is learned from the bare groove, or from the quietest
                   moments if the needle lands mid-song. "Song" = the 75 Hz pilot standing 3.5x over the channels
                   beside it for 0.2 s; it ends 0.4 s after that falls under 2.5x. Discs without the pilot are treated
                   as one long gap
  volume cap       the level control boosts at most 4x (listen.py: 60x)

Measured on those takes against the original soundtrack: closer than listen.py during the
song, and quieter on bare groove. Weak needle landings are sometimes taken for a gap and turned down.
"""
import collections
import sys
import threading
import time
import wave

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, sosfilt, sosfiltfilt

import listen
from capture import FS, decode, write_wav

BLOCK = 512
LOOK = 192                      # samples of look-ahead for click filling (20 ms)
NFFT, HOP = 512, 128
FLOOR = 10 ** (-18 / 20)                    # noise reduction between songs
PILOT_ON, PILOT_OFF = 3.5, 2.5              # 75 Hz over the empty channels beside it
ON_BLOCKS, HOLD_S = 4, 0.4                  # blocks above ON to call it a song; seconds below OFF to end it
EQ = np.zeros(NFFT // 2 + 1)                # gain per bin (18.8 Hz apart)
EQ[7:54] = [
    0.7929, 0.7676, 0.7171, 0.7086, 0.7736, 0.8019, 0.8640, 1.0157, 1.1319, 1.0889,
    1.2362, 1.4386, 1.5546, 1.6714, 1.7431, 1.6537, 1.6079, 1.6678, 1.4658, 1.3626,
    1.5930, 1.5956, 1.5016, 1.5990, 1.5422, 1.4854, 1.4229, 1.2083, 1.3172, 1.7638,
    2.1020, 2.0111, 2.1667, 2.4614, 1.9215, 1.3892, 1.6134, 1.8580, 1.7832, 1.7451,
    1.5261, 1.3827, 1.4450, 1.5512, 1.5293, 1.6730, 1.6765]
GAIN_MAX = 4.0


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


class Cleaner:
    """Feed raw samples (volts / 2.5, centred on the bias) in any block size; get cleaned audio back, delayed."""

    def __init__(self):
        full, _ = listen.chain()
        self.pre = full[:-2]                              # listen.py's chain without its 1 kHz low-pass (2 sections)
        self.zpre = np.zeros((len(self.pre), 2))
        self.click_hp = butter(4, 1200, "highpass", fs=FS, output="sos")
        self.zc = np.zeros((2, 2))
        self.buf = np.zeros(LOOK)                         # samples waiting for their look-ahead
        self.high = np.zeros(LOOK)
        self.hist = collections.deque(maxlen=40)          # recent high-band medians (about 2 s)
        self.clicks = 0
        self.win = np.sqrt(np.hanning(NFFT + 1)[:-1])
        self.inq = np.zeros(NFFT - HOP)
        self.ola = np.zeros(NFFT)
        self.noise = None                                 # noise power per bin
        self.learned = 0
        self.recent = collections.deque(maxlen=int(3 * FS / HOP))
        self.prev = np.zeros(NFFT // 2 + 1)
        self.pil = np.zeros(int(0.4 * FS))                # last 0.4 s of raw input, for the pilot
        self.down_s = 0.0
        self.gain = 3.0
        self.song, self.count, self.left = False, 0, 0.0
        self.blend = 0.0                                  # 0: between songs, 1: song playing

    def _pilot_and_level(self, raw):
        self.pil = np.concatenate([self.pil, raw])[-len(self.pil):]
        t = np.arange(len(self.pil)) / FS
        w = np.hanning(len(self.pil))
        p = [abs(np.sum(self.pil * w * np.exp(-2j * np.pi * f * t))) for f in (66.5, 74.67, 90.0)]
        return p[1] / (0.5 * (p[0] + p[2]) + 1e-12)

    def _detect(self, ratio, seconds):
        self.count = self.count + 1 if ratio > PILOT_ON else 0
        if self.count >= ON_BLOCKS:
            self.song, self.left = True, HOLD_S
        elif self.song:
            self.left = HOLD_S if ratio > PILOT_OFF else self.left - seconds
            self.song = self.left > 0

    def _declick(self, x):
        h, self.zc = sosfilt(self.click_hp, x, zi=self.zc)
        n = len(self.buf)
        self.buf = np.concatenate([self.buf, x])
        self.high = np.concatenate([self.high, np.abs(h)])
        env = maximum_filter1d(self.high, 5)
        self.hist.append(float(np.median(env[n:])) if len(env) > n else 0.0)
        thr = max(4.0 * float(np.median(self.hist)), 0.008)            # 0.008 = 20 mV
        out_n = len(self.buf) - LOOK
        bad = maximum_filter1d((env > thr).astype(np.uint8), 19).astype(bool)
        e = np.diff(np.concatenate([[0], bad.astype(int), [0]]))
        for s, f in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
            if s < out_n and f - s <= int(0.005 * FS) and s > 48 and f < len(self.buf) - 48:
                _ar_fill(self.buf, s, f, ctx=min(160, s, len(self.buf) - f))
                self.high[s:f] = 0.0
                self.clicks += 1
        out, self.buf, self.high = self.buf[:out_n].copy(), self.buf[out_n:], self.high[out_n:]
        return out

    def _denoise(self, y, ratio):
        self.inq = np.concatenate([self.inq, y])
        out = []
        while len(self.inq) >= NFFT:
            X = np.fft.rfft(self.inq[:NFFT] * self.win)
            P = np.abs(X) ** 2
            self.recent.append(P)
            level = float(np.sqrt(np.sum(P) * 2 / NFFT / np.sum(self.win ** 2)))
            down = level > 0.006                                      # about 15 mV in band: the needle is on the disc
            self.down_s = self.down_s + HOP / FS if down else 0.0
            if down and ratio < 1.6 and 0.3 < self.down_s < 3.0:      # bare groove before the song
                self.learned += 1
                a = max(0.03, 1.0 / self.learned)
                self.noise = P.copy() if self.noise is None else (1 - a) * self.noise + a * P
            noise = self.noise
            if noise is None and len(self.recent) > 40:               # dropped mid-song: the quietest fifth stands in
                noise = 1.8 * np.percentile(np.array(self.recent), 20, axis=0)
            if noise is None:
                G = np.ones_like(P)
            else:
                post = P / (noise + 1e-12)
                prio = 0.96 * self.prev / (noise + 1e-12) + 0.04 * np.maximum(post - 1.0, 0.0)
                G = np.maximum(prio / (1.0 + prio), FLOOR)
                G = uniform_filter1d(G, 3)
            self.prev = G * G * P
            self.blend += float(np.clip(self.song - self.blend, -HOP / (0.2 * FS), HOP / (0.2 * FS)))
            G = self.blend + (1.0 - self.blend) * G
            self.ola += np.fft.irfft(X * G * EQ) * self.win
            out.append(self.ola[:HOP] / 2.0)                          # sqrt-Hann at 4x overlap sums to 2
            self.ola = np.concatenate([self.ola[HOP:], np.zeros(HOP)])
            self.inq = self.inq[HOP:]
        return np.concatenate(out) if out else np.zeros(0)

    def __call__(self, raw):
        ratio = self._pilot_and_level(raw)
        self._detect(ratio, len(raw) / FS)
        y, self.zpre = sosfilt(self.pre, raw, zi=self.zpre)
        y = self._declick(y)
        if not len(y):
            return y
        y = self._denoise(y, ratio)
        if not len(y):
            return y
        rms = float(np.sqrt(np.mean(y * y))) + 1e-6
        a = min(1.0, len(y) / (FS * listen.GAIN_TAU))
        self.gain = (1 - a) * self.gain + a * min(GAIN_MAX, listen.TARGET_RMS / rms)
        return np.clip(y * self.gain, -1.0, 1.0)


def sim(src, dst):
    with wave.open(src) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(float) / 32767
    c, out, worst = Cleaner(), [], 0.0
    t0 = time.perf_counter()
    for i in range(0, len(x), BLOCK):
        t = time.perf_counter()
        out.append(c(x[i:i + BLOCK]))
        worst = max(worst, time.perf_counter() - t)
    y = np.concatenate(out)
    write_wav(dst, y, FS)
    print(f"{len(x) / FS:.1f} s processed in {time.perf_counter() - t0:.1f} s; slowest block {1000 * worst:.1f} ms of "
          f"{1000 * BLOCK / FS:.0f} ms available; {c.clicks} clicks filled; noise learned from {c.learned * HOP / FS:.1f} s of bare groove")


def live(port, seconds, out):
    import serial
    import sounddevice as sd
    rate, up = listen.open_output(sd)
    fifo, lock, state = collections.deque(), threading.Lock(), {"held": np.zeros(0, dtype=np.float32)}

    def callback(outdata, frames, t, status):
        buf = state["held"]
        with lock:
            while len(buf) < frames and fifo:
                buf = np.concatenate([buf, fifo.popleft()])
        n = min(frames, len(buf))
        outdata[:n, 0] = buf[:n]
        outdata[n:, 0] = 0.0
        state["held"] = buf[n:]

    c, raw, heard, tail, pend = Cleaner(), [], [], b"", np.zeros(0)
    with serial.Serial(port, 500000, timeout=0.5) as s, \
            sd.OutputStream(samplerate=rate, channels=1, dtype="float32", blocksize=BLOCK * up, latency="high", callback=callback):
        time.sleep(2.0)
        s.reset_input_buffer()
        print(f"recording {seconds:.0f} s from {port}; monitor at {rate} Hz, cleaned")
        t0 = time.time()
        try:
            while time.time() - t0 < seconds:
                chunk = s.read(1024)
                raw.append(chunk)
                v, _ = decode(tail + chunk)
                tail = (tail + chunk)[-1:]
                pend = np.concatenate([pend, (v * 5.0 / 1023.0 - 2.5) / 2.5])
                while len(pend) >= BLOCK:
                    y = c(pend[:BLOCK]).astype(np.float32)
                    pend = pend[BLOCK:]
                    if len(y):
                        heard.append(y)
                        with lock:
                            fifo.append(np.repeat(y, up).astype(np.float32) if up > 1 else y)
        except KeyboardInterrupt:
            print("stopped early")
        time.sleep(0.5)
    v, lost = decode(b"".join(raw))
    volts = v * 5.0 / 1023.0
    print(f"{len(v)} samples ({len(v) / FS:.1f} s), mean {volts.mean():.3f} V, lost {lost}, {c.clicks} clicks filled")
    x = volts - volts.mean()
    write_wav(out, x / 2.5, FS)
    hp = sosfiltfilt(butter(2, 300, "highpass", fs=FS, output="sos"), x)
    write_wav(out.replace(".wav", ".hp.wav"), hp / (np.abs(hp).max() or 1), FS)
    write_wav(out.replace(".wav", ".live.wav"), np.concatenate(heard) if heard else np.zeros(1), FS)
    print(f"wrote {out}, .hp.wav and .live.wav")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    if sys.argv[1] == "--sim":
        sim(sys.argv[2], sys.argv[3])
    else:
        live(sys.argv[1], float(sys.argv[2]), sys.argv[3])
