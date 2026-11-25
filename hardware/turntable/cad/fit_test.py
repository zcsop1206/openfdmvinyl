"""
fit_test.py - a minimal fit-test plate for the Nov 24 spine reprint: coupons cut from the real parts (Nov 24 2026).

    python fit_test.py        # writes stl_spine/fit_*.stl; then  python ../print/make_plates.py spine 9

Every coupon is the real part's CAD intersected with a small box or sector around one fit, in the part's own print
orientation, so it prints the fit exactly as the full part will. Print it before the full plates:

  fit_tower       spine tower, top 13.5 mm: the V bore on the sleeve, the T-bar slot, the M3x12 clamp pilot and its
                  spot face. Drop the sleeve coupon in, the T-pin through it, turn the screw onto the pin: no play.
  fit_sleeve      sleeve, lower 18 mm: the pin's V and the screw slot.
  fit_servo       spine servo pocket, top 9.5 mm, with the crush ribs and the port-end wall: press the servo in from
                  above; it should go in firmly by hand and not rock.
  fit_hub         arm hub: the M3x8 button head's counterbore (drive it in), the funnel, the skirt's dowel holes.
  fit_stub        20 mm of the arm's stub, for the carriage slice.
  fit_carriage    8 mm slice of the counterweight's top: the stub channel with its friction tongue.
  fit_carriage_nuts  the same slice's lower block: one column of acorn-nut holes (push a nut in: snug, slides with a
                  push). The lightening window splits the slice into these two pieces.
  fit_rim         40 deg of the platter rim: one notch and its ramped ledge.
  fit_ring        16 deg of the clamp ring with one lug: drop it through the notch on a scrap of 1 mm disc, slide it
                  along the ledge until it pinches.
  fit_weight      centre weight: the spindle bore and one acorn-nut pocket.
  damper_cup, damper_skirt (whole, already small): the cup over the sleeve coupon; the skirt on two filament dowels
                  in the hub coupon.
"""
import math
from pathlib import Path

from build123d import Align, Box, Polygon, Pos, Rot, export_stl, extrude

import spine as S                         # installs the headshell, steady and rim_clamp parts
import turntable as T

SD, RC = S.SD, S.RC
MIN = (Align.CENTER, Align.CENTER, Align.MIN)
OUT = Path(__file__).parent / "stl_spine"


def box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, z0) * Box(x1 - x0, y1 - y0, z1 - z0, align=MIN)


def sector(r0, r1, a0, a1, z0, z1, n=48):
    a = [math.radians(a0 + (a1 - a0) * k / n) for k in range(n + 1)]
    pts = [(r1 * math.cos(t), r1 * math.sin(t)) for t in a] + [(r0 * math.cos(t), r0 * math.sin(t)) for t in a[::-1]]
    return Pos(0, 0, z0) * extrude(Polygon(*pts, align=None), z1 - z0)


def one(shape, name):
    sol = shape.solids()
    assert len(sol) >= 1, name
    s = max(sol, key=lambda x: x.volume)
    bb = s.bounding_box()
    return Pos(-(bb.min.X + bb.max.X) / 2, -(bb.min.Y + bb.max.Y) / 2, -bb.min.Z) * s   # on the bed, centred


def coupons():
    c = {}
    sp = S.spine()
    ax = T.ARM_D
    c["fit_tower"] = sp & box(ax - 15, ax + 15, -15, 15, SD.CLAMP_Z - 5.0, T.TOWER_H + 1)
    x_near = -(T.SERVO_L - T.SHAFT_FROM_END) - T.FIT
    x_far = T.SHAFT_FROM_END + T.FIT
    yw = T.SERVO_W / 2 + T.FIT
    c["fit_servo"] = sp & box(x_near - 4.5, x_far + 4.5, -yw - 4.5, yw + 4.5, T.FLOOR + T.POCKET_H - 9.5, T.FLOOR + T.POCKET_H + 1)
    c["fit_sleeve"] = SD.sleeve() & box(-6, 6, -6, 6, 0, 18.0)
    arm = T.arm_side(T.arm())                          # mirrored to the side it plays on (Nov 25)
    c["fit_hub"] = arm & box(-11, 11, -11, 11, -1, T.HUB_T + 1)
    c["fit_stub"] = arm & box(-T.STUB_L, -T.STUB_L + 20, -T.BEAM / 2 - 0.5, T.BEAM / 2 + 0.5, -1, T.BEAM + 1)
    car = (Rot(X=180) * (T.arm_side(T.weight()) & box(-4, 4, -20, 20, -40, 40))).solids()               # roof down; the window
    car = sorted(car, key=lambda x: x.bounding_box().min.Z)                                  # splits the slice in two
    c["fit_carriage"], c["fit_carriage_nuts"] = car[0], car[-1]
    p = S.open_platter()
    trav = RC.TRAVEL + 10
    c["fit_rim"] = Rot(X=180) * (p & sector(83.0, 93.0, -trav, 6.0, -1, 30))                      # top face down
    c["fit_ring"] = Rot(X=180) * (RC.ring() & sector(80.0, 95.0, -8.0, 8.0, -20, 5))               # flange down
    w = RC.centre_weight()
    c["fit_weight"] = w & box(-5.5, RC.PUCK_RINGS[1][0] + 4.5, -5.5, 5.5, -1, RC.PUCK_H + 1)
    return {k: one(v, k) for k, v in c.items()}


def main():
    for name, s in coupons().items():
        export_stl(s, str(OUT / f"{name}.stl"))
        bb = s.bounding_box()
        print(f"  {name:13s} {bb.size.X:5.1f} x {bb.size.Y:5.1f} x {bb.size.Z:5.1f} mm  {s.volume / 1000:5.2f} cm3")


if __name__ == "__main__":
    main()
