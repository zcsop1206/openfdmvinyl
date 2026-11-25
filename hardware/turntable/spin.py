"""
spin.py - run the platter at playing speed until Ctrl-C.

    python spin.py COM5 [motor_id] [--rpm 40] [--ramp 1.0] [--seconds 20] [--log pos.csv]

Velocity mode, ramped up over --ramp seconds in small steps (a hard start is safe for the record,
but a ramp is kinder to the horn screws). Prints the measured speed every 2 s from the
unwrapped position, so you can check 40 RPM without a strobe. Ctrl-C ramps down, turns torque off and
puts the servo back in position mode. --log writes every position read (time s, unwrapped position in turns)
as fast as the bus allows, to look for speed ripple faster than the 2 s average (the 2.8 Hz pitch wobble on the
v3 disc, Nov 24 2026).

Needs lerobot 0.7 with the Feetech extra, like servo_speed_test.py. Run capture.py for the cartridge in a
second terminal: the servo board and the Arduino are separate USB devices.
"""

import argparse
import time

from lerobot.motors import Motor, MotorNormMode, SerialMotorsBus

STEPS = 4096


def rpm_to_raw(rpm):
    return int(round(rpm / 60.0 * STEPS))   # Goal_Velocity is steps/s


def ramp(bus, a, b, seconds, n=20):
    for k in range(1, n + 1):
        bus.write("Goal_Velocity", "tt", rpm_to_raw(a + (b - a) * k / n), normalize=False, num_retry=3)   # a dropped reply aborted take 3 (Nov 22)
        time.sleep(seconds / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("port")
    ap.add_argument("motor_id", nargs="?", type=int, default=1)
    ap.add_argument("--rpm", type=float, default=40.0)
    ap.add_argument("--ramp", type=float, default=1.0)
    ap.add_argument("--seconds", type=float, default=0.0, help="stop by itself after this long (0 = until Ctrl-C)")
    ap.add_argument("--log", default=None, help="CSV of every position read: time_s, turns (no sleep between reads)")
    a = ap.parse_args()

    bus = SerialMotorsBus(port=a.port, motors={"tt": Motor(a.motor_id, "sts3215", MotorNormMode.DEGREES)})
    bus.connect(handshake=False)
    try:
        bus.disable_torque(num_retry=2)
        bus.set_operating_mode("velocity", "tt")
        bus.write("Goal_Velocity", "tt", 0, normalize=False)
        bus.write("Torque_Enable", "tt", 1, normalize=False)
        print(f"motor {a.motor_id}: {bus.read('Present_Voltage', 'tt', normalize=False) / 10:.1f} V; "
              f"ramping to {a.rpm} RPM, Ctrl-C to stop")
        ramp(bus, 0.0, a.rpm, a.ramp)
        last_p, last_t, turns, t0 = None, None, 0.0, time.perf_counter()
        log = open(a.log, "w") if a.log else None
        t_start, total = t0, 0.0
        t_end = t0 + a.seconds if a.seconds > 0 else float("inf")
        while time.perf_counter() < t_end:
            p, t = bus.read("Present_Position", "tt", normalize=False, num_retry=3), time.perf_counter()
            if last_p is not None:
                d = p - last_p
                d += -STEPS if d > STEPS / 2 else (STEPS if d < -STEPS / 2 else 0)
                turns += d / STEPS
                total += d / STEPS
            last_p = p
            if log:
                log.write(f"{t - t_start:.5f},{total:.6f}\n")
            if t - t0 >= 2.0:
                print(f"  {turns / (t - t0) * 60:6.2f} RPM")
                turns, t0 = 0.0, t
            if not log:
                time.sleep(0.01)
        print("time up, stopping")
        if log:
            log.close()
        ramp(bus, a.rpm, 0.0, a.ramp)
    except KeyboardInterrupt:
        print("stopping")
        try:
            ramp(bus, a.rpm, 0.0, a.ramp)
        except Exception:
            pass
    finally:
        try:
            bus.write("Goal_Velocity", "tt", 0, normalize=False)
        except Exception:
            pass
        bus.disable_torque(num_retry=2)
        bus.set_operating_mode("position", "tt")
        bus.disconnect(disable_torque=True)


if __name__ == "__main__":
    main()
