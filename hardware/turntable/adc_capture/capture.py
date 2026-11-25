"""
capture.py - record adc_capture.ino's stream to a WAV.

    python capture.py COM3 20 take1.wav

Writes take1.wav (raw, DC removed, 9615 Hz, 16-bit) and take1.hp.wav (300 Hz high-pass,
normalized, for listening). Prints the level and how many samples were lost to resync.
"""

import sys
import time
import wave

import numpy as np
import serial

FS = 16_000_000 / 128 / 13      # 9615.38 Hz, the UNO's free-running ADC rate


def decode(raw: bytes) -> tuple:
    b = np.frombuffer(raw, dtype=np.uint8)
    hi_idx = np.nonzero(((b[:-1] & 0x80) != 0) & ((b[1:] & 0x80) == 0))[0]
    v = ((b[hi_idx].astype(np.int32) & 0x1F) << 5) | (b[hi_idx + 1] & 0x1F)
    lost = len(b) // 2 - len(v)
    return v, lost


def write_wav(path, x, fs):
    x = np.clip(x, -1, 1)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(round(fs)))
        w.writeframes((x * 32767).astype("<i2").tobytes())


def main():
    port, seconds, out = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    with serial.Serial(port, 500000, timeout=0.5) as s:
        time.sleep(2.0)                 # the UNO resets when the port opens
        s.reset_input_buffer()
        t0 = time.time()
        chunks = []
        while time.time() - t0 < seconds:
            chunks.append(s.read(4096))
    v, lost = decode(b"".join(chunks))
    volts = v * 5.0 / 1023.0
    print(f"{len(v)} samples ({len(v) / FS:.1f} s), mean {volts.mean():.3f} V, "
          f"min {volts.min():.3f} V, max {volts.max():.3f} V, lost {lost}")
    if volts.min() < 0.05 or volts.max() > 4.95:
        print("CLIPPING: lower the groove amplitude or add a divider")
    x = volts - volts.mean()
    write_wav(out, x / 2.5, FS)
    from scipy.signal import butter, sosfiltfilt
    hp = sosfiltfilt(butter(2, 300, "highpass", fs=FS, output="sos"), x)
    peak = np.abs(hp).max()
    write_wav(out.replace(".wav", ".hp.wav"), hp / peak if peak > 0 else hp, FS)
    print(f"wrote {out} and {out.replace('.wav', '.hp.wav')}")


if __name__ == "__main__":
    main()
