"""
headshell.py - the headshell, designed around the measured CKANDAY cartridge (cartridge.py), and the pad it bolts to
on the end of the arm. Nov 22 2026.

    python headshell.py        # fit checks, print check, writes headshell_fit.png and stl_spine/headshell.stl

Why two parts: the cartridge's top (its rear block) sits 9.9 mm above the stylus tip, which puts the holder 3.5 to
7.9 mm below the arm's underside. The arm prints flat on that underside, so a one-piece holder would hang in the air
(supports inside the socket, on the very faces that set the stylus height). As its own part the headshell prints
top-down: the ceiling the cartridge bears on and the top face that bears on the arm both come off the bed.

How it holds the cartridge, feature by feature (cartridge.py frame: x back from the red front, z up from the
bottom, stylus down):
  height      a stepped ceiling: z 8.5 over the rails, z 9.4 over the raised rear block. The tracking force pushes
              the cartridge up into it, so the stylus height is set by these two faces.
  fore-aft    the step in the ceiling (the rear block's front ramp, 0.2 mm clear) and the rear wall behind the
              block's overhang (0.15 mm clear); the tab F sits in a slot through the rear wall (0.15 a side).
  sideways    side walls 0.1 mm clear of the 9.5 mm body, and the tab slot.
  retention   a sprung tongue in each side wall presses a 0.15 mm bump on the black body (x 19 to 25, where the
              body is black from top to bottom). The only downward loads are the cartridge's 2 g and handling.
  free        open at the front and below z 5.0: the red stylus (top 4.6) slides off forward, the front notch H
              is reachable, and the pins and lead clips are clear behind and below.
The window G is not used: its inside could not be measured, and a hook in a plain hole holds nothing downward.

On the arm: a flat pad replaces the slotted plate. Two M2x8 button heads from the fastener box go down through the
pad into the headshell's bosses, and a 22 mm length of 1.75 mm filament in matching grooves (pad underside,
headshell top) sets the angle, so the offset angle does not depend on screw-hole play.
"""
import math
from pathlib import Path

import numpy as np
from build123d import Align, Box, Cylinder, Plane, Polygon, Pos, Rot, extrude, export_stl

import turntable as T
import cartridge as K

HERE = Path(__file__).parent
MIN = (Align.CENTER, Align.CENTER, Align.MIN)

CLR_SIDE = 0.05          # body to side wall (0.10 printed Nov 22 fit well but a little loose; slightly tighter wanted, Nov 23)
WALL = 1.6
WALL_Z0 = 5.0            # side walls end here: above the red's top (4.6)
STEP_CLR = 0.1           # ceiling step ahead of the rear block's ramp (0.2 until Nov 23)
REAR_CLR = 0.10          # overhang back face to the rear wall (0.15 until Nov 23)
TAB_CLR = 0.15
REAR_WALL = 1.6
REAR_WALL_Z0 = 7.0       # rear wall only above this: the upper lead clips (pins at z 5.2) pass under it
TONGUE_X = (19.0, 25.0)  # sprung tongues, on the black body between the red (ends x 18) and the rear face (27)
TONGUE_T, TONGUE_TOP = 0.8, 8.8
BUMP = 0.25              # tongue bump interference with the body (0.12 until Nov 23, 0.18 printed Nov 24: still loose)
SLIT = 0.6
TOP_Z = T.CART_H + K.TIP_Z   # 12.9: headshell top = arm underside (stylus 13.4 below it)
BOLT_X, BOLT_Y = 12.0, 8.4   # M2 bosses beside the walls, in cartridge.py x
BOSS_R, BOLT_HOLE = 3.3, 2.4   # through-bolted (Nov 22: self-tapping M2 pilots printed too tight); 2.4 prints ~2.1
NUT_AF, NUT_T, NUT_TRAP_AF, NUT_TRAP_T = 4.0, 1.6, 4.4, 1.9   # M2 nut, and the hex trap at the boss's foot
BOLT_L = 12.0                # M2x12 button head: pad 3 + boss 7.9, then the nut
DOWEL_D, DOWEL_L = 1.75, 16.0
DOWEL_X = (4.0, 26.0)        # the arm pad's groove (printed with the arm)
HS_GROOVE_X = (4.0, 20.0)    # the headshell's groove, shorter since the pocket was pitched (its ceiling rises behind x 21)
GROOVE_W, GROOVE_D = 2.05, 0.95   # prints ~1.8 for the 1.75 dowel
PAD_T = T.SHELL_T            # 3.0
PAD_HOLE = 2.6   # M2 clearance
LIFT_X, LIFT_W = (-4.0, 8.0), 14.0     # finger lift: along the cartridge axis, and out from the pad
LIP_T, LIP_H = 1.6, 5.0               # upturned lip on its outer edge, above the tab
X_REAR = K.OVERHANG_X + REAR_CLR + REAR_WALL     # 30.25
HALF_W = K.W / 2 + CLR_SIDE + WALL               # 6.45
# Pitch (Nov 23): with the stylus loaded, the cartridge's flat black underside (0.5 mm above the tip) sagged
# onto the record. The pocket now holds the cartridge pitched nose-down about the stylus tip until the red stylus's
# sloped underside lies parallel to the record. The tip stays where it was, so the arm, its pad, the bolts and the
# stylus height are unchanged. The lowest part of the body is now that red underside, 1.9 mm above the tip, and the
# back of the black body rises 2.5 mm.
RAKE = math.degrees(math.atan2(K.RED_FRONT_LIFT, K.RED_BOT_L))    # 5.78 deg


def seat():
    """cartridge.py frame -> headshell frame: the cartridge pitched nose-down by RAKE about its stylus tip."""
    return Pos(K.TIP_X, 0, K.TIP_Z) * Rot(Y=-RAKE) * Pos(-K.TIP_X, 0, -K.TIP_Z)


def KtoC():
    """cartridge.py frame -> the arm's cartridge frame (stylus at the origin, +x back along the cartridge, z = 0
    the arm's underside)."""
    return Pos(-K.TIP_X, 0, -T.CART_H - K.TIP_Z)


def box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, z0) * Box(x1 - x0, y1 - y0, z1 - z0, align=MIN)


def prism_yz(pts, x0, x1):
    """A (y, z) profile extruded along +x (counter-clockwise first)."""
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    pts = pts if area > 0 else pts[::-1]
    return Pos(x0, 0, 0) * extrude(Plane.YZ * Polygon(*pts, align=None), x1 - x0)


def headshell(bumps=True):
    """Headshell frame: the flat top (TOP_Z), bosses, bolt holes, nut traps and dowel groove are in cartridge.py
    coordinates as before; the walls and the pocket are built around the cartridge and pitched by seat()."""
    hw_in = K.W / 2 + CLR_SIDE
    h = box(0.0, X_REAR, -HALF_W, HALF_W, WALL_Z0, TOP_Z + 4.0)      # tall: still reaches the top once pitched
    # pocket: low ceiling over the rails, high ceiling over the rear block, rear wall with the tab slot
    x_step = K.STEP_X - K.STEP_RAMP - STEP_CLR
    h -= box(-1.0, x_step, -hw_in, hw_in, WALL_Z0 - 1, K.H_MID)
    h -= box(x_step, K.OVERHANG_X + REAR_CLR, -hw_in, hw_in, WALL_Z0 - 1, K.H_REAR)
    h -= box(K.OVERHANG_X - 0.5, X_REAR + 1, -K.TAB_W / 2 - TAB_CLR, K.TAB_W / 2 + TAB_CLR,
             WALL_Z0 - 1, K.H_REAR)                     # tab slot, open below so the tab rises straight in
    h -= box(K.OVERHANG_X, X_REAR + 1, -hw_in, hw_in, WALL_Z0 - 1, REAR_WALL_Z0)      # rear wall stops at z 7
    # tongues: slit at each end through the wall, outer half of the wall removed so 0.8 mm flexes
    t0, t1 = TONGUE_X
    for s in (-1, 1):
        for xa, xb in ((t0 - SLIT, t0), (t1, t1 + SLIT)):
            h -= box(xa, xb, *sorted((s * hw_in, s * (HALF_W + 1))), WALL_Z0 - 1, TONGUE_TOP)
        h -= box(t0, t1, *sorted((s * (hw_in + TONGUE_T), s * (HALF_W + 1))), WALL_Z0 - 1, TONGUE_TOP)
        if bumps:   # lead-in from the wall's lower edge, then 0.15 mm into the body
            yb = K.W / 2 - BUMP
            prof = [(s * hw_in, WALL_Z0), (s * yb, WALL_Z0 + 0.6), (s * yb, WALL_Z0 + 1.2), (s * hw_in, WALL_Z0 + 1.6)]
            h += prism_yz(prof, t0 + 0.5, t1 - 0.5)
    h = seat() * h
    h = h & box(-10.0, X_REAR + 10.0, -HALF_W - 10.0, HALF_W + 10.0, -10.0, TOP_Z)   # flat top on the pad
    for s in (-1, 1):
        h += Pos(BOLT_X, s * BOLT_Y, WALL_Z0) * Cylinder(BOSS_R, TOP_Z - WALL_Z0, align=MIN)
    # bolt holes straight through, with a hex trap for the nut at each boss's foot (it opens upward in print,
    # top face down, so no overhang); then the dowel groove on top
    for s in (-1, 1):
        h -= Pos(BOLT_X, s * BOLT_Y, WALL_Z0 - 1.0) * Cylinder(BOLT_HOLE / 2, TOP_Z - WALL_Z0 + 2.0, align=MIN)
        h -= Pos(BOLT_X, s * BOLT_Y, WALL_Z0 - 1.0) * hex_prism(NUT_TRAP_AF, NUT_TRAP_T + 1.0)
    h -= box(HS_GROOVE_X[0], HS_GROOVE_X[1], -GROOVE_W / 2, GROOVE_W / 2, TOP_Z - GROOVE_D, TOP_Z + 1)
    solids = h.solids()
    assert len(solids) == 1, ("headshell came out in pieces", [round(x.volume) for x in solids])
    return solids[0]


def dowel():
    x0 = (HS_GROOVE_X[0] + HS_GROOVE_X[1] - DOWEL_L) / 2
    return Pos(x0 + DOWEL_L / 2, 0, TOP_Z) * Rot(Y=90) * Cylinder(DOWEL_D / 2, DOWEL_L)


def hex_prism(af, h):
    """A hexagonal prism, across-flats af, height h, base on z = 0."""
    from build123d import RegularPolygon
    return extrude(RegularPolygon(af / math.sqrt(3), 6), h)


def nuts():
    """The M2 nuts seated in their traps (cartridge.py frame)."""
    from build123d import Compound
    return Compound(children=[(Pos(BOLT_X, s * BOLT_Y, WALL_Z0) * (hex_prism(NUT_AF, NUT_T) -
                               Cylinder(0.8, NUT_T, align=MIN))).solids()[0] for s in (-1, 1)])


def bolts():
    """M2x12 button heads (ISO 7380: head 3.5 x 1.3) from the pad's top down through the bosses into the nuts,
    cartridge.py frame."""
    out = []
    for s in (-1, 1):
        b = Pos(BOLT_X, s * BOLT_Y, TOP_Z + PAD_T - BOLT_L) * Cylinder(1.0, BOLT_L, align=MIN)
        b += Pos(BOLT_X, s * BOLT_Y, TOP_Z + PAD_T) * Cylinder(1.75, 1.3, align=MIN)
        out.append(b)
    from build123d import Compound
    return Compound(children=[x.solids()[0] for x in out])


def pad():
    """The arm's pad in the arm's cartridge frame (T.HEADSHELL_PAD): flat on the bed with the arm, the two M2
    clearance holes, the dowel groove on its underside, the finger lift, and a bridge to the beam's end."""
    bx, by = beam_end_in_cartridge_frame()
    c = KtoC()
    x0, x1 = -1.0, X_REAR - K.TIP_X - 1.5           # stops short of the rear wall: the leads go up behind it
    y_lo, y_hi = -HALF_W - 3.0, HALF_W + 3.0
    p = box(x0, x1, y_lo, y_hi, 0, PAD_T)
    p += box(min(x0, bx - 6), max(x1 - 6, bx + 4), *sorted((0.0, by + math.copysign(5.0, by))), 0, PAD_T)
    # finger lift on the outside (+y lands outboard of the stylus): a tab to get a fingertip under, 13 mm above
    # the record, and an upturned lip along its outer edge so the finger can't slide off. The lip is a vertical
    # wall on the tab, so it prints with the arm (flat) with no supports.
    lx0, lx1 = LIFT_X
    p += box(lx0, lx1, y_hi - 0.01, y_hi + LIFT_W, 0, PAD_T)
    p += box(lx0, lx1, y_hi + LIFT_W - LIP_T, y_hi + LIFT_W, 0, PAD_T + LIP_H)
    for s in (-1, 1):
        p -= Pos(BOLT_X - K.TIP_X, s * BOLT_Y, 0) * Cylinder(PAD_HOLE / 2, PAD_T, align=MIN)
    p -= c * box(DOWEL_X[0], DOWEL_X[1], -GROOVE_W / 2, GROOVE_W / 2, TOP_Z - 1, TOP_Z + GROOVE_D)
    return p


def beam_end_in_cartridge_frame():
    """Where the arm's beam ends (arm frame (ARM_L - 24, 0)) in the cartridge frame."""
    sgn = T.headshell_sign()
    a = math.radians(180 + sgn * T.ARM_BETA)
    dx, dy = (T.ARM_L - 24.0) - T.ARM_L, 0.0
    return dx * math.cos(-a) - dy * math.sin(-a), dx * math.sin(-a) + dy * math.cos(-a)


# ---- the arm and the assembly use these -------------------------------------------------------------------------
def cartridge_C():
    k = K.cartridge()
    return KtoC() * seat() * (k["black"] + k["red"])


def headshell_C():
    return KtoC() * headshell()


def red_C():
    return KtoC() * seat() * K.red()


def install():
    """Point turntable.py at the measured cartridge, the headshell and the new pad."""
    T.HEADSHELL_PAD = pad
    T.cartridge = cartridge_C
    T.cart_mount = headshell_C
    T.cartridge_stylus = red_C


# ---- checks ---------------------------------------------------------------------------------------------------
def common(a, b):
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    c = BRepAlgoAPI_Common(a.wrapped, b.wrapped); c.Build()
    g = GProp_GProps(); BRepGProp.VolumeProperties_s(c.Shape(), g)
    return abs(g.Mass())


def clips():
    """Headshell lead clips stand-in: 2.0 mm OD sleeves, 6 mm long, on each pin from 0.3 behind the rear face."""
    out = None
    for z, c in K.PINS:
        for s in (-1, 1):
            cl = Pos(K.LEN + 0.3 + 3.0, s * c / 2, z) * Rot(Y=90) * Cylinder(1.0, 6.0)
            out = cl if out is None else out + cl
    return out


def fit_checks(verbose=True):
    from motion import dist
    S = seat()
    k = K.cartridge()
    black0, red0 = k["black"], k["red"]
    black, red = S * black0, S * red0
    cart = black + red
    hs, hs_nb = headshell(), headshell(bumps=False)
    bad = []

    def say(msg, ok=True):
        if verbose:
            print(("  " if ok else "  PROBLEM: ") + msg)
        if not ok:
            bad.append(msg)

    print(f"headshell for the CKANDAY: {X_REAR:.2f} x {2 * HALF_W:.1f} x {TOP_Z - WALL_Z0:.1f} mm, "
          f"{hs.volume / 1000:.2f} cm3 (~{hs.volume * 1.24e-3 * 0.9:.1f} g)")
    ov = common(cart, hs_nb)
    say(f"seated: cartridge | headshell (bumps off) overlap {ov:.3f} mm3", ov < 0.01)
    rails = dist(S * (black0 & box(5, 20, -K.W, K.W, K.H_MID - 0.2, K.H_MID + 1)), hs_nb)
    block = dist(S * (black0 & box(K.STEP_X + 0.1, K.OVERHANG_X, -K.W, K.W, K.H_REAR - 0.2, K.H_REAR + 1)), hs_nb)
    say(f"height datum: rails to the low ceiling {rails:.2f} mm, rear block to the high ceiling {block:.2f} mm "
        f"-> stylus {T.CART_H:.1f} mm below the arm's underside", rails < 0.01 and block < 0.01)
    # play: slide the cartridge in each direction until it touches
    def play(vec):
        for d in np.arange(0.02, 1.5, 0.02):
            if common(S * Pos(*(np.array(vec) * d)) * black0, hs_nb) > 0.001:
                return d - 0.02
        return None
    pf, pr, ps = play((-1, 0, 0)), play((1, 0, 0)), play((0, 1, 0))
    say(f"play: forward {pf:.2f}, back {pr:.2f}, sideways {ps:.2f} mm (forward stop = ceiling step, back = rear "
        f"wall, sideways = walls and tab slot)", None not in (pf, pr, ps) and max(pf, pr, ps) <= 0.3)
    bump = common(cart, hs) - ov
    say(f"retention: tongue bumps press {bump:.2f} mm3 into the body ({BUMP} mm each side)", bump > 0.1)
    # insertion: pushed up along the cartridge's own height, bumps off (the tongues spring)
    ins = max(common(S * Pos(0, 0, -d) * (black0 + red0), hs_nb) for d in np.arange(0.0, 10.01, 0.5))
    say(f"insertion along the cartridge's height from 10 mm below: worst overlap {ins:.3f} mm3 (tongues aside)", ins < 0.01)
    # tongue strain and force
    L = TONGUE_TOP - WALL_Z0
    I = (TONGUE_X[1] - TONGUE_X[0]) * TONGUE_T ** 3 / 12
    d = BUMP
    strain, force = 3 * d * TONGUE_T / (2 * L ** 2), 3 * 3500 * I * d / L ** 3
    say(f"tongues: {TONGUE_T} x {L:.1f} mm, deflect {d:.2f} mm: strain {strain * 100:.1f} % (PLA yield ~2.5), "
        f"{force:.1f} N each; held by ~{2 * force * 0.3:.1f} N of friction against 0.02 N of weight", strain < 0.022)   # 0.25 mm bumps: ~2.1 %, under PLA yield (~2.5 %)
    # the red stylus slides off forward
    rv = max(common(S * Pos(-dx, 0, 0) * red0, hs) for dx in np.arange(0, 20.1, 1.0))
    say(f"stylus pull-off, 20 mm forward: overlap {rv:.3f} mm3", rv < 0.01)
    # lead clips
    cl = S * clips()
    cv = common(cl, hs)
    say(f"lead clips (2.0 OD) on the pins: overlap with the headshell {cv:.3f} mm3 (with the cartridge's own "
        f"overhang {common(cl, black):.1f} mm3: its underside z {K.OVERHANG_Z} is from a photo, and the rear photo "
        f"shows slots over the upper pins; not a headshell issue)", cv < 0.01)
    # pad, bolts and dowel
    p = pad()
    hsC = headshell_C()
    ov_p = common(p, hsC)
    bl = KtoC() * bolts()
    ov_b = common(bl, KtoC() * headshell()) + common(KtoC() * nuts(), KtoC() * headshell())
    nut_engaged = (WALL_Z0 + NUT_T) - (TOP_Z + PAD_T - BOLT_L)               # thread through the nut, mm
    poke = WALL_Z0 - (TOP_Z + PAD_T - BOLT_L)                                # bolt end below the boss foot
    rec = (TOP_Z + PAD_T - BOLT_L) - K.TIP_Z                                 # bolt end above the record
    nut_wall = BOSS_R - NUT_TRAP_AF / math.sqrt(3)                           # boss wall at the trap's corners
    dw = KtoC() * dowel()
    ov_d = common(dw, p) + common(dw, hsC)
    say(f"on the pad: pad | headshell {ov_p:.3f} mm3, dowel in both grooves {ov_d:.3f} mm3; M2x{BOLT_L:.0f} bolts and "
        f"nuts | headshell {ov_b:.3f} mm3 (through-holes {BOLT_HOLE}, traps {NUT_TRAP_AF} AF); bolt through the nut "
        f"{nut_engaged:.1f} mm of its {NUT_T}, end {poke:.1f} mm below the boss and {rec:.1f} mm above the record; "
        f"boss wall at the trap corners {nut_wall:.2f} mm",
        ov_p < 0.01 and ov_d < 0.01 and ov_b < 0.01 and nut_engaged >= NUT_T - 0.01 and rec > 3.0 and nut_wall > 0.6)
    ang = math.degrees(math.atan((GROOVE_W - DOWEL_D) / DOWEL_L))
    say(f"angle: dowel play {GROOVE_W - DOWEL_D:.2f} mm over {DOWEL_L:.0f} mm -> +-{ang:.2f} deg on the offset")
    # pitch: the red underside lies flat, and how far each part of the body sits above the record at the tip
    a, b = (S * Pos(0, 0, K.RED_FRONT_LIFT)).position.Z, (S * Pos(K.RED_BOT_L, 0, 0)).position.Z
    red_body = K.prism_xz([(0.0, K.RED_FRONT_LIFT), (K.RED_BOT_L, 0.0), (K.RED_TOP_L, K.RED_H), (0.0, K.RED_H)],
                          -K.W / 2, K.W / 2)                             # the red housing, without cantilever and tip
    body = black + S * red_body
    low = body.bounding_box().min.Z - K.TIP_Z
    say(f"pitch {RAKE:.2f} deg nose-down about the tip: red underside z {a:.2f} / {b:.2f} (flat); lowest point of the "
        f"body {low:.2f} mm above the record (was {-K.TIP_Z:.2f}); back of the body up {(S * Pos(K.LEN, 0, 0)).position.Z:.2f} mm",
        abs(a - b) < 0.01 and low > 1.5)
    # the pocket's ceiling under the flat top and under the dowel groove
    top_slab = box(-1, X_REAR + 1, -HALF_W, HALF_W, TOP_Z - 0.6, TOP_Z + 1)
    groove_skin = dist(cart, box(HS_GROOVE_X[0], HS_GROOVE_X[1], -GROOVE_W / 2, GROOVE_W / 2, TOP_Z - GROOVE_D, TOP_Z + 1))
    say(f"top skin: cartridge to the flat top {dist(cart, Pos(0, 0, 0.6) * top_slab):.2f} mm, to the dowel groove floor "
        f"{groove_skin:.2f} mm", groove_skin >= 0.6)
    return bad


def print_check():
    """The headshell prints top face down: list faces that face the bed from above it (overhangs)."""
    flip = Rot(X=180) * Pos(0, 0, -TOP_Z)
    h = flip * headshell()
    bb = h.bounding_box()
    h = Pos(0, 0, -bb.min.Z) * h
    worst = []
    for f in h.faces():
        n = f.normal_at()
        c = f.center()
        if n.Z < -0.71 and c.Z > 0.2:
            worst.append((f.area, c.Z, (round(c.X, 1), round(c.Y, 1))))
    worst.sort(reverse=True)
    print(f"  print (top down): {len(worst)} downward faces above the bed" +
          ("".join(f"\n    {a:5.1f} mm2 at z {z:.1f} {xy}" for a, z, xy in worst[:6])))
    return h


def figure(path):
    """Side and end sections through the cartridge in the headshell, and an exploded view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import trimesh
    k = {n: seat() * v for n, v in K.cartridge().items()}
    hs = headshell()

    def sec(shape, origin, normal, axes):
        v, f = shape.tessellate(0.01, 0.1)
        m = trimesh.Trimesh(np.array([tuple(q) for q in v]), np.array(f), process=False)
        s = m.section(plane_origin=origin, plane_normal=normal)
        return [] if s is None else [d[:, axes] for d in s.discrete]
    fig, axs = plt.subplots(2, 2, figsize=(14, 8), dpi=110)
    cuts = [((0, 0, 0), (0, 1, 0), [0, 2], "side section on the centreline"),
            ((0, 3.25, 0), (0, 1, 0), [0, 2], "side section through the upper pins and a tongue"),
            ((22.0, 0, 0), (1, 0, 0), [1, 2], "end section at x 22 (tongues, rear-block step)"),
            ((28.9, 0, 0), (1, 0, 0), [1, 2], "end section at x 28.9 (rear wall, tab slot)")]
    for ax, (o, n, axes, title) in zip(axs.flat, cuts):
        for shape, col in ((hs, "#d9a441"), (k["black"], "#2b2b29"), (k["red"], "#c0392b"), (seat() * clips(), "#9a9a92")):
            for loop in sec(shape, o, n, axes):
                ax.fill(loop[:, 0], loop[:, 1], color=col, alpha=0.9, lw=0.4, ec="k")
        ax.axhline(K.TIP_Z, color="#888", lw=0.6, ls=":")
        ax.set_aspect("equal"); ax.set_title(title, fontsize=9); ax.tick_params(labelsize=7)
    fig.suptitle(f"CKANDAY cartridge in the headshell (gold), pitched {RAKE:.1f} deg so the red underside is flat; grey = lead "
                 "clips; dotted = record surface at the tip",
                 fontsize=10)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    install()
    bad = fit_checks()
    h = print_check()
    export_stl(h, str(HERE / "stl_spine" / "headshell.stl"))
    figure(HERE / "headshell_fit.png")
    print(f"wrote stl_spine/headshell.stl and headshell_fit.png; {len(bad)} problem(s)")
