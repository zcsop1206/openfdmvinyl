"""
Tonearm balance: counterweight fill for a tracking force, and the static roll of the unipivot arm.

    python balance.py

With the stylus on the record the arm rests on two points, the needle tip and the stylus, so it can only
roll about the line through them. It settles where its centre of gravity hangs straight below that line;
the roll angle is atan(sideways CG offset / CG depth below the line). Anything off to one side (the
headshell offset, the leads on the tie tabs) tilts the stylus; the counterweight cup sits WEIGHT_Y to the
other side to cancel it. Masses: PLA 1.24 g/cm3 at 80 % fill (from the slicer's grams),
cartridge 2.1 g (CZ-800 spec) in the printed mount (turntable.cart_mount), leads 1 g inside the beam.
"""
import math

import numpy as np
from build123d import Pos, Rot

import turntable as T

RHO, FILL = 1.24e-3, 0.80        # 0.80 from the Bambu slice of the arm plate (Nov 21)
COIN_AIM = 4.0
M_CART, M_LEADS = 2.1, 1.0       # CZ-800 net weight (Adelcom); weigh yours when it arrives


def mass_cg(shape, fill=FILL):
    c = shape.center()
    return shape.volume * RHO * fill, np.array([c.X, c.Y, c.Z])


def solve(force_g, arm=None, carriage=None):
    """Coins (g) and carriage x for a tracking force; CG and roll. Arm frame: pivot axis at x = y = 0."""
    sgn = T.headshell_sign()
    ma, ca = arm or mass_cg(T.arm())
    mw, cw0 = carriage or mass_cg(T.weight())
    cw0 = cw0 + [-60.0, 0, -(30 - 2 - T.BEAM - 0.5) - 0.3]
    cart = Pos(T.ARM_L, 0, 0) * Rot(Z=180 + sgn * T.ARM_BETA) * T.cartridge()
    cc = mass_cg(cart, 1)[1]
    leads = np.array([70.0, 0.0, T.BEAM / 2])        # inside the beam tunnel
    crad = Pos(T.ARM_L, 0, 0) * Rot(Z=180 + sgn * T.ARM_BETA) * T.cart_mount()
    mcr, ccr = mass_cg(crad)
    fixed = [(ma, ca), (M_CART, cc), (M_LEADS, leads), (mcr + 0.6, ccr)]   # mount + three M2x6 (0.2 g each)
    best = None
    for xw in np.arange(-22.0, -T.STUB_L + 10, -1.0):     # carriage anywhere on the stub: aim for COIN_AIM g of
        cw = cw0 + [xw - cw0[0], 0, 0]                   # coins, so there is mass to trim with
        coin = np.array([xw, T.WEIGHT_Y, cw[2] - 4])     # coins sit low in the cup
        moment = sum(m * c[0] for m, c in fixed) + mw * cw[0]
        mc = (moment - force_g * T.ARM_L) / (-coin[0])
        if mc >= 0 and (best is None or abs(mc - COIN_AIM) < abs(best[1] - COIN_AIM)):
            best = (xw, mc, cw, coin)
    xw, mc, cw, coin = best
    parts = fixed + [(mw, cw), (mc, coin)]
    m = sum(p[0] for p in parts)
    g = sum(p[0] * p[1] for p in parts) / m
    # roll axis: needle tip (0, 0, CONE_H) to stylus tip (ARM_L, 0, -CART_H)
    a, b = np.array([0, 0, T.CONE_H]), np.array([T.ARM_L, 0, -T.CART_H])
    u = (b - a) / np.linalg.norm(b - a)
    v = g - a - ((g - a) @ u) * u                         # CG offset from the axis, perpendicular to it
    roll = math.degrees(math.atan2(v[1], -v[2]))
    return dict(force=force_g, carriage_x=float(xw), coins=float(mc), mass=float(m), cg=g,
                depth=float(-v[2]), side=float(v[1]), roll=roll)


def main():
    arm, car = mass_cg(T.arm()), mass_cg(T.weight())
    print(f"arm {arm[0]:.1f} g, carriage {car[0]:.1f} g, cartridge {M_CART} g (CZ-800 spec), leads {M_LEADS} g; "
          f"cup offset WEIGHT_Y = {T.WEIGHT_Y} mm")
    for f in (4.0, 5.0, 6.0):                              # the cartridge's rated range
        s = solve(f, arm, car)
        print(f"  {f:.0f} g force: carriage at x {s['carriage_x']:.0f}, coins {s['coins']:4.1f} g, arm total "
              f"{s['mass']:4.1f} g; CG {s['depth']:4.1f} mm below the pivot-stylus line, {s['side']:+.2f} mm "
              f"sideways -> roll {s['roll']:+.1f} deg")


if __name__ == "__main__":
    main()
