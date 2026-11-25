"""
listen.py - capture.py with a live monitor: record the UNO's stream and hear a cleaned copy while it plays.

    python listen.py COM15 50 take17.wav          # needs: pip install sounddevice

Writes exactly what capture.py writes (take17.wav raw, take17.hp.wav for listening), so the analysis is unchanged, plus
take17.live.wav, the cleaned stream as it was heard. Only one program can hold the port: run this instead of capture.py,
not beside it.

The live chain, causal and block by block (Nov 24 2026):
  120 Hz high-pass (6th order)    rumble and the 27.9 Hz drive line go (the disc's music starts at 120 Hz)
  75 Hz notch (Q 8)               the pilot tone
  notches at 60 Hz multiples      mains hum and its harmonics up to 900 Hz (Q 30)
  1 kHz low-pass (4th order)      hiss above the music band (the groove carries nothing past ~850 Hz)
  slow gain                       holds the level steady; it rises gently in quiet passages
Played at the ADC's 9615 Hz if the sound device takes it, else upsampled 5x and played at 48 kHz (0.16 % sharp).
"""
import collections
import sys
import threading
import time

import numpy as np
import serial
from scipy.signal import butter, iirnotch, sosfilt, sosfilt_zi, tf2sos

from capture import FS, decode, write_wav

BLOCK = 512
PILOT_HZ = 75.0                  # the song discs' pilot tone (out/song/libets_delay/*.manifest.json)
TARGET_RMS, GAIN_MAX, GAIN_TAU = 0.15, 60.0, 2.0     # level held at 0.15 rms; gain settles over ~2 s


def chain():
    secs = [butter(6, 120, "highpass", fs=FS, output="sos")]
    b, a = iirnotch(PILOT_HZ, 8.0, fs=FS)                       # the pilot, +-0.5 % of platter speed inside the notch
    secs.append(tf2sos(b, a))
    for k in range(1, 16):
        f = 60.0 * k
        if f < 900:
            b, a = iirnotch(f, 30.0, fs=FS)
            secs.append(tf2sos(b, a))
    secs.append(butter(4, 1000, "lowpass", fs=FS, output="sos"))
    sos = np.vstack(secs)
    return sos, sosfilt_zi(sos) * 0.0


def open_output(sd):
    """(stream, upsample factor). Try the ADC rate first; most Windows (MME) devices resample it themselves."""
    for rate, up in ((int(round(FS)), 1), (48000, 5)):
        try:
            sd.check_output_settings(samplerate=rate, channels=1, dtype="float32")
            return rate, up
        except Exception:
            continue
    raise RuntimeError("no output rate accepted (tried 9615 and 48000 Hz)")


def main():
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("listen.py needs sounddevice: pip install sounddevice (in the venv you record with)")
    port, seconds, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    sos, zi = chain()
    rate, up = open_output(sd)
    fifo = collections.deque()
    lock = threading.Lock()
    state = {"held": np.zeros(0, dtype=np.float32)}

    def callback(outdata, frames, t, status):
        buf = state["held"]
        with lock:
            while len(buf) < frames and fifo:
                buf = np.concatenate([buf, fifo.popleft()])
        n = min(frames, len(buf))
        outdata[:n, 0] = buf[:n]
        outdata[n:, 0] = 0.0                                   # underrun: silence rather than a stall
        state["held"] = buf[n:]

    raw, live = [], []
    gain = 10.0
    tail = b""
    with serial.Serial(port, 500000, timeout=0.5) as s, \
            sd.OutputStream(samplerate=rate, channels=1, dtype="float32", blocksize=BLOCK * up, latency="high",
                            callback=callback):
        time.sleep(2.0)                                        # the UNO resets when the port opens
        s.reset_input_buffer()
        print(f"recording {seconds:.0f} s from {port}; monitor at {rate} Hz{' (5x up)' if up > 1 else ''}. Ctrl-C stops early.")
        t0 = time.time()
        try:
            while time.time() - t0 < seconds:
                chunk = s.read(4096)
                raw.append(chunk)
                v, _ = decode(tail + chunk)
                tail = (tail + chunk)[-1:]                     # a high byte whose low byte is in the next chunk
                if not len(v):
                    continue
                x = (v * 5.0 / 1023.0 - 2.5) / 2.5
                y, zi = sosfilt(sos, x, zi=zi)
                rms = float(np.sqrt(np.mean(y * y))) + 1e-6
                a = min(1.0, len(y) / (FS * GAIN_TAU))
                gain = (1 - a) * gain + a * min(GAIN_MAX, TARGET_RMS / rms)
                y = np.clip(y * gain, -1.0, 1.0).astype(np.float32)
                live.append(y)
                if up > 1:
                    y = np.repeat(y, up).astype(np.float32)        # monitor only; the saved files stay at the ADC rate
                with lock:
                    fifo.append(y)
        except KeyboardInterrupt:
            print("stopped early")
        time.sleep(0.5)                                        # let the last block play
    # the same files as capture.py, from the whole raw stream
    v, lost = decode(b"".join(raw))
    volts = v * 5.0 / 1023.0
    print(f"{len(v)} samples ({len(v) / FS:.1f} s), mean {volts.mean():.3f} V, min {volts.min():.3f} V, "
          f"max {volts.max():.3f} V, lost {lost}")
    if volts.min() < 0.05 or volts.max() > 4.95:
        print("CLIPPING: lower the groove amplitude or add a divider")
    x = volts - volts.mean()
    write_wav(out, x / 2.5, FS)
    from scipy.signal import sosfiltfilt
    hp = sosfiltfilt(butter(2, 300, "highpass", fs=FS, output="sos"), x)
    peak = np.abs(hp).max()
    write_wav(out.replace(".wav", ".hp.wav"), hp / peak if peak > 0 else hp, FS)
    write_wav(out.replace(".wav", ".live.wav"), np.concatenate(live) if live else np.zeros(1), FS)
    print(f"wrote {out}, {out.replace('.wav', '.hp.wav')} and {out.replace('.wav', '.live.wav')}")


if __name__ == "__main__":
    main()
