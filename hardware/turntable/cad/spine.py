"""
spine.py - the minimal first build that carries forward into the briefcase (Nov 22 2026).

    python spine.py          # writes stl_spine/*.stl and prints the checks

Same turntable.py parts, with the briefcase's arm geometry and only what a first play needs:
  spine      servo pocket + beam + pivot tower + wire anchor, feet cut to +-45 mm (no outrigger, tray, rest,
             shell bosses); later it bolts under the briefcase deck
  arm        135 mm effective length, pivot 119.8 mm from the spindle, 28.1 deg offset (+-0.68 deg tracking
             error), rear stub 45 mm so the counterweight stays within 40 mm of the pivot
  platter    open platter: rim, eight spokes over the ribs, and the hub (horn slots, spindle socket); the
             record sits on the rim and spokes
  small      weight, sleeve, shims, spindle as in turntable.py; the headshell (headshell.py) on a pad at the arm's end
  Nov 24     steady.py: V-bored tower with a clamp screw, V-bored sleeve, acorn-nut carriage, damper cup and hub
             skirt (dowel holes in the hub); rim_clamp.py: twist-lock ring, notches and ledges in the platter rim
turntable.py is not edited: its module constants are overridden here before any part is built.
"""
import math
import shutil
from pathlib import Path

import numpy as np
from build123d import Align, Box, Cylinder, Pos, Rot, export_stl

import turntable as T

# ---- briefcase arm geometry -------------------------------------------------------------------------------------
T.ARM_L, T.ARM_D = 135.0, 119.8
_r = np.linspace(T.GROOVE_R[0], T.GROOVE_R[1], 600)
_phi = np.degrees(np.arcsin((_r * _r + T.ARM_L ** 2 - T.ARM_D ** 2) / (2 * _r * T.ARM_L)))
T.ARM_BETA = float((_phi.max() + _phi.min()) / 2)    # offset that balances the error at both ends of the groove
T.STUB_L = 45.0
T.TIE_X = (T.ARM_L - 24.0 - 6.0,)                   # strain-relief tab just inside the shorter beam's end
import headshell as HS                    # noqa: E402  measured cartridge, its headshell, the arm's pad
HS.install()
import steady as SD                       # noqa: E402  rigid pivot, acorn-nut carriage, damper (Nov 24)
SD.install()
import rim_clamp as RC                    # noqa: E402  twist-lock ring for the disc edge (Nov 24)
FOOT_HALF = 45.0                          # feet cut back from +-70 (tape the spine down for the first plays)
OUT = Path(__file__).parent / "stl_spine"


def tracking_error():
    r = np.linspace(T.GROOVE_R[0], T.GROOVE_R[1], 300)
    phi = np.degrees(np.arcsin((r * r + T.ARM_L ** 2 - T.ARM_D ** 2) / (2 * r * T.ARM_L)))
    return float(np.max(np.abs(phi - T.ARM_BETA)))


def spine():
    b = T.base()
    x_near = -(T.SERVO_L - T.SHAFT_FROM_END) - T.FIT
    big = 400.0
    cut = Pos(0, FOOT_HALF + big / 2, 0) * Box(big * 2, big, big)            # feet beyond +-45
    cut += Pos(0, -FOOT_HALF - big / 2, 0) * Box(big * 2, big, big)
    cut += Pos(x_near - 4 - big / 2, 0, 0) * Box(big, big * 2, big)           # outrigger
    (tx0, ty0), (tx1, ty1), _ = T.ELECTRONICS_BAY
    tray_x1 = T.ARM_D - 7.0                                                   # up to the tower foot bar's face
    cut += Pos((tx0 + tray_x1) / 2, (ty0 - 1 + -10.5) / 2, 0) * Box(tray_x1 - tx0, -10.5 - (ty0 - 1), 60,
                                                                  align=(Align.CENTER, Align.CENTER, Align.MIN))
    cut += T.rest_frame() * Box(T.REST_LEN + 8, T.REST_W + 8, 300)            # arm rest post (built for the old arm)
    # and the strip that joined the post to the tower, with its root fillets; the tower, its foot bar and the
    # beam are kept whole
    s_len = T.REST_D + T.REST_LEN / 2 + 4
    strip = Pos(T.ARM_D, 0, 0) * Rot(Z=T.REST_ANGLE) * Pos(s_len / 2, 0, 0) * Box(
        s_len, 10 + 2 * 6.5, T.FOOT_H + 3.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    beam_x0 = T.SHAFT_FROM_END + T.FIT + 4
    strip -= Pos(T.ARM_D, 0, 0) * Cylinder(T.TOWER_OD / 2, 300, align=(Align.CENTER, Align.CENTER, Align.MIN))
    strip -= Pos(T.ARM_D, 0, 0) * Box(14, 2 * FOOT_HALF, T.FOOT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    strip -= Pos((beam_x0 + T.ARM_D) / 2, 0, 0) * Box(T.ARM_D - beam_x0, 20, T.FOOT_H,
                                                      align=(Align.CENTER, Align.CENTER, Align.MIN))
    cut += strip
    # slimming: hollow the tower below the T-pin floor, with a 45 deg conical roof (no bridge); open the
    # servo floor under the horn half from below, leaving a 2.5 mm skin (a 20 mm bridge) for the cover to sit on
    r_in = T.TOWER_OD / 2 - 2.4
    roof = T.FLOOR_Z - 2.0
    cut += Pos(T.ARM_D, 0, 0) * Cylinder(r_in, roof - r_in, align=(Align.CENTER, Align.CENTER, Align.MIN))
    from build123d import Cone
    cut += Pos(T.ARM_D, 0, roof - r_in) * Cone(r_in, 0.01, r_in, align=(Align.CENTER, Align.CENTER, Align.MIN))
    cut += Pos(1.25, 0, 0) * Box(13.5, 20.0, T.SEAT - 2.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    s = b - cut
    s = max(s.solids(), key=lambda x: x.volume)                               # drop any islands
    s = servo_grip(s)                                                         # crush ribs, port-end wall (Nov 24)
    return SD.tower_mods(s)                                                   # V bore and clamp screw (Nov 24)


RIB_W, RIB_BITE = 1.2, 0.25      # crush rib width, and how far it stands into the servo's outline
PORT_WALL_T, PORT_WALL_Z0 = 4.0, T.SEAT + 2.0   # port-end wall, above the cover plate (the cable leaves below z 9.5);
                                                # 4 mm: flush with the ends of the side walls, so the joint has no inside corner


def servo_grip(s):
    """Nov 24: the servo sat loose every way in the pocket (0.4 mm a side; held with clay). Crush ribs on the side
    walls and both end walls stand RIB_BITE into the servo's outline and shave as it is pressed in, so the fit is
    tight whatever the print tolerance. The port end, open since Nov 21, gets a wall back above the cover plate
    (the plugs hang below it and the cable leaves under the bridge, below z 9.5). Each rib has a 45 deg lead-in at
    the top. All vertical: prints as the spine does, upright."""
    from build123d import Plane, Polygon, extrude
    x_far = T.SHAFT_FROM_END + T.FIT
    x_near = -(T.SERVO_L - T.SHAFT_FROM_END) - T.FIT
    yw = T.SERVO_W / 2 + T.FIT
    z0, z1 = T.FLOOR + 1.0, T.FLOOR + T.POCKET_H
    p = T.FIT + RIB_BITE                                       # rib depth from the wall
    prof = Polygon((0, z0), (p, z0), (p, z1 - p - 0.6), (0, z1 - 0.6), align=None)   # (depth, z), lead-in at the top
    rib = Pos(0, -RIB_W / 2, 0) * extrude(Plane.XZ * prof, -RIB_W)                 # depth along +x, width along y
    wall = Pos(x_near - PORT_WALL_T / 2, 0, PORT_WALL_Z0) * Box(PORT_WALL_T, 2 * yw + 0.02, z1 - PORT_WALL_Z0,
                                                             align=(Align.CENTER, Align.CENTER, Align.MIN))
    s = s + wall
    ribs = None
    for (x, y, ang) in [(x, sy * yw, -90 * sy) for x in (-26.0, 2.0) for sy in (-1, 1)] + \
                       [(x_far, y, 180) for y in (-7.0, 7.0)] + [(x_near, y, 0) for y in (-7.0, 7.0)]:
        r = Pos(x, y, 0) * Rot(Z=ang) * rib
        ribs = r if ribs is None else ribs + r
    s = s + ribs
    sol = s.solids()
    assert len(sol) == 1, ("servo grip split the spine", [round(x.volume) for x in sol])
    # 0.3 mm fillets at each rib's root (vertical edges, so they face sideways as printed), which takes the shear as
    # the servo goes in. The pocket's own corners stay sharp (a fit).
    sites = [(x, sy * yw) for x in (-26.0, 2.0) for sy in (-1, 1)] + [(x_far, y) for y in (-7.0, 7.0)] +             [(x_near, y) for y in (-7.0, 7.0)]
    s, _ = T.fillet_inside(sol[0], 0.3, keep=lambda p: z0 - 0.1 < p.Z < z1 + 0.1 and
                           min(math.hypot(p.X - a, p.Y - b) for a, b in sites) < RIB_W / 2 + 0.3)
    return s


def open_platter():
    """Hub, rim and eight spokes of the full platter: the windows between the spokes cut out of it. (Intersecting
    with a hub + rim + spokes mask returned only the spindle socket's core once the socket became a through-hole,
    Nov 22: an OCC boolean failure near the axis. This cut stays 22 mm off the axis.)"""
    p = T.platter()
    MIN = (Align.CENTER, Align.CENTER, Align.MIN)
    windows = Pos(0, 0, -1) * (Cylinder(T.PLAT_R - 10, 102, align=MIN) - Cylinder(22.0, 102, align=MIN))
    for k in range(8):
        windows -= Rot(Z=22.5 + 45 * k) * Pos((T.PLAT_R - 5) / 2, 0, -1) * Box(T.PLAT_R - 5, 6, 102, align=MIN)
    solids = (p - windows).solids()
    assert len(solids) == 1 and solids[0].volume > 40000, ("open platter came out wrong", [round(x.volume) for x in solids])
    return RC.add_to_platter(solids[0])                                       # clamp ring notches and ledges (Nov 24)


def main():
    OUT.mkdir(exist_ok=True)
    print(f"arm {T.ARM_L} mm, pivot {T.ARM_D} mm, offset {T.ARM_BETA:.2f} deg: max tracking error {tracking_error():.2f} deg")
    for r in T.GROOVE_R:
        off, ok = T.groove_drag_ok(r)
        print(f"  r {r:.0f}: cartridge {off:.2f} deg off the groove, groove runs {'pins -> stylus' if ok else 'stylus -> pins (BACKWARDS)'}")
        assert off < 1.0 and ok, "arm drawn for the wrong spin direction"
    parts = {"spine": spine(), "platter_open": Rot(X=180) * open_platter(), "arm": T.arm_side(T.arm()),
             "weight": Rot(X=180) * T.arm_side(T.weight()), "peg": T.peg(), "spindle": T.spindle()}   # key down (Nov 22)
    parts.update({f"shim_{t:.0f}": T.shim(t) for t in T.SHIMS})
    parts["headshell"] = Rot(X=180) * Pos(0, 0, -HS.TOP_Z) * T.arm_side(HS.headshell())  # top face down; symmetric
    for k, d in enumerate(T.SPINDLE_SET, 1):                                 # disc-hole fit set (Nov 22)
        parts[f"spindle_{round(d * 100)}"] = T.spindle(d, marks=k)
    parts["damper_cup"] = Pos(0, 0, -SD.CUP_FLOOR[0]) * SD.cup()           # floor down (Nov 24)
    parts["damper_skirt"] = Rot(X=180) * T.arm_side(SD.skirt())             # washer down; symmetric
    parts["clamp_ring"] = Rot(X=180) * RC.ring()                             # flange down
    parts["centre_weight"] = RC.centre_weight()                              # as it sits (Nov 24)
    for name, shape in parts.items():
        export_stl(shape, str(OUT / f"{name}.stl"))
        bb = shape.bounding_box()
        print(f"  {name:12s} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:5.1f} mm  {shape.volume / 1000:6.1f} cm3")


if __name__ == "__main__":
    main()
