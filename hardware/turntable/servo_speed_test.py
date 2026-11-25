"""
servo_speed_test.py - how fast and how steadily one STS3215 turns in wheel (velocity) mode.

    python servo_speed_test.py COM5 [motor_id] [--rpm 33.33 40 45 ...]

The motor must be OUT OF THE ARM with its horn free (or with the platter on it for a
loaded run): velocity mode turns endlessly. For each target RPM it spins up, settles 2 s,
then logs Present_Position for 8 s. Speed comes from the unwrapped position, not the
coarse Present_Velocity register. It prints mean RPM, error against the target, and
speed ripple over 0.25 s windows (what you would hear as wow), then writes
servo_speed_<id>.csv with every sample. Torque is off and the mode is back to position
on exit.

Needs lerobot 0.7 with the Feetech extra (pip install "lerobot[feetech]"; 0.7 was used) for the servo bus.
"""

import argparse
import csv
import time

import numpy as np
from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

STEPS = 4096


def rpm_to_raw(rpm):
    return int(round(rpm / 60.0 * STEPS))   # Goal_Velocity is steps/s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("motor_id", nargs="?", type=int, default=1)
    ap.add_argument("--rpm", type=float, nargs="+", default=[33.33, 38, 42, 45, 50])
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--hold", type=float, default=8.0)
    a = ap.parse_args()

    bus = SerialMotorsBus(port=a.port, motors={"tt": Motor(a.motor_id, "sts3215", MotorNormMode.DEGREES)})
    bus.connect(handshake=False)
    rows = []
    try:
        bus.disable_torque(num_retry=2)
        bus.set_operating_mode("velocity", "tt")
        bus.write("Goal_Velocity", "tt", 0, normalize=False)
        bus.write("Torque_Enable", "tt", 1, normalize=False)
        print(f"motor {a.motor_id}: voltage {bus.read('Present_Voltage', 'tt', normalize=False) / 10:.1f} V")
        for target in a.rpm:
            bus.write("Goal_Velocity", "tt", rpm_to_raw(target), normalize=False)
            time.sleep(a.settle)
            t0 = time.perf_counter()
            ts, pos, load = [], [], []
            while (t := time.perf_counter() - t0) < a.hold:
                ts.append(t)
                pos.append(bus.read("Present_Position", "tt", normalize=False))
                load.append(bus.read("Present_Load", "tt", normalize=False))
            ts, pos = np.array(ts), np.array(pos, dtype=float)
            # unwrap the 12-bit position (one turn = 4096 steps)
            d = np.diff(pos)
            d[d > STEPS / 2] -= STEPS
            d[d < -STEPS / 2] += STEPS
            turns = np.concatenate([[0.0], np.cumsum(d)]) / STEPS
            rpm_mean = (turns[-1] - turns[0]) / (ts[-1] - ts[0]) * 60.0
            # speed over 0.25 s windows
            edges = np.arange(0.0, ts[-1], 0.25)
            idx = np.searchsorted(ts, edges)
            w = [(turns[j] - turns[i]) / (ts[j] - ts[i]) * 60.0
                 for i, j in zip(idx[:-1], idx[1:]) if j > i + 1]
            w = np.array(w)
            ripple = 100.0 * (w.max() - w.min()) / 2.0 / rpm_mean if len(w) else float("nan")
            rate = len(ts) / ts[-1]
            print(f"target {target:6.2f} rpm: mean {rpm_mean:6.2f} ({100 * (rpm_mean / target - 1):+5.2f} %), "
                  f"ripple +-{ripple:4.2f} % (0.25 s windows), load up to {max(abs(x) for x in load) / 10:.0f} %, "
                  f"{rate:.0f} reads/s")
            rows += [(target, t, p, l) for t, p, l in zip(ts, pos, load)]
    finally:
        try:
            bus.write("Goal_Velocity", "tt", 0, normalize=False)
        except Exception:
            pass
        bus.disable_torque(num_retry=2)
        bus.set_operating_mode("position", "tt")
        bus.disconnect(disable_torque=True)
        with open(f"servo_speed_{a.motor_id}.csv", "w", newline="") as fh:
            cw = csv.writer(fh)
            cw.writerow(["target_rpm", "t_s", "position_raw", "load_raw"])
            cw.writerows(rows)


if __name__ == "__main__":
    main()
