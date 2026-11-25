"""
steady.py - a steadier unipivot for the spine build (Nov 24 2026).

    python steady.py          # checks: pivot fit, damper fit, balance and roll with the acorn-nut carriage

Why: every take from the spine turntable shows a 2.8 Hz loudness pulse on the pilot (54-119 % swing), the arm
"tilts sometimes and needs adjusting", "rocks side to side" and "wanders or shifts on the pin" (Nov 24).
With the arm off, the T-pin wiggled in the printed sleeve and the sleeve wiggled in the tower. Three changes,
keeping the unipivot (it keeps the drive's vibration out of the arm):

  rigid pivot   The tower bore gets a 90 deg V on its -y side; the sleeve's pin bore gets one too. An M3x12
                button head, self-tapped sideways through the tower wall at CLAMP_Z, passes through a slot in the
                sleeve and presses the pin into the sleeve's V; the pin pushes the sleeve into the tower's V. One
                screw, no play in either fit. The shims still carry the weight and set the height; the sleeve's slot
                covers the whole shim range.
  carriage      The counterweight holds M3 acorn nuts in hex holes through it (2 per hole, 6 holes), so nothing
                can roll about. The nuts sit lower than the coins did, which puts the arm's CG deeper under the
                pivot-stylus line (static tilt changes less for a given sideways shift). Trim roll by sliding nuts
                along their holes.
  damper        A cup on the tower top around the sleeve holds a liquid (dish soap, oil, honey: not grease, whose
                yield stress would drag the arm sideways). A ring skirt pinned under the arm's hub with two
                filament dowels dips into it with 1 mm gaps both sides. Roll and pitch squeeze the liquid through
                the gaps (damped); tracking only shears it (negligible drag). The fill height tunes it.

Frames: turntable frame (spindle at the origin, desk z = 0) for the tower, cup and screw; arm frame (pivot axis
at x = y = 0, arm underside z = 0) for the skirt and carriage; part frames as noted.
"""
import math

import numpy as np
from build123d import Align, Box, Cylinder, Plane, Polygon, Pos, Rot, extrude

import turntable as T

MIN = (Align.CENTER, Align.CENTER, Align.MIN)

# ---- rigid pivot ---------------------------------------------------------------------------------------------
SLEEVE_R = T.PEG_D / 2            # 5.0
BORE_CLR = 0.35                   # tower bore radius over the sleeve on the screw side (the old 10.7 bore)
V_APEX = SLEEVE_R * math.sqrt(2)  # tower V: flats at +-45 deg tangent to the sleeve, apex 7.07 from the axis
CLAMP_Z = 60.0                    # clamp screw height above the desk (tower top 68.5, sleeve 48.4-77.5 on the 4 mm shim)
PILOT_R = 1.25                    # M3 self-tapping pilot (2.5, the M3 tap size; as the pivot cup's)
SCREW_L = 12.0                    # M3x12 button head
FLAT_Y = 11.2                     # spot face for the screw head on the tower's +y side
PIN_R = 0.6                       # 1.1-1.2 mm T-pin
PIN_V_APEX = PIN_R * math.sqrt(2)
PIN_SLOT_W, PIN_SLOT_TOP = 2.4, 1.3   # pin bore: V on -y, open to +y (prints ~2.0 wide: holes print 0.2-0.4 under)
SLEEVE_SLOT_W = 3.4               # screw passes through the sleeve wall (no thread there)
SHIM_RANGE = (1.0, 7.0)           # shim stacks the sleeve slot allows (1, 2, 4 mm shims: 1 to 7)


def tower_bore_profile(z0, h):
    """The new bore around the tower axis (tower frame, x = y = 0): half circle BORE on +y (screw side), V on -y."""
    r = SLEEVE_R + BORE_CLR
    half = Cylinder(r, h, align=MIN) & Pos(0, r, 0) * Box(2 * r + 1, 2 * r, h, align=MIN)
    v = extrude(Polygon((-V_APEX, 0.01), (0, -V_APEX), (V_APEX, 0.01), align=None), h)
    return Pos(0, 0, z0) * (half + v)


def tower_mods(s):
    """Spine part -> the spine with the V bore, the clamp screw pilot and its spot face."""
    c = Pos(T.ARM_D, 0, 0)
    h = T.TOWER_H - T.FLOOR_Z
    s = s + c * Pos(0, 0, T.FLOOR_Z) * Cylinder(T.TOWER_BORE / 2 + 0.05, h, align=MIN)      # fill the old bore
    s = s - c * tower_bore_profile(T.FLOOR_Z, h + 1)
    s = s - c * Pos(0, 0, T.FLOOR_Z) * Box(T.SLOT_L, T.SLOT_W, h + 1, align=MIN)            # T-bar slot again
    s = s - c * Pos(0, 6.5, CLAMP_Z) * Rot(Z=90) * T.teardrop_x(PILOT_R, 13.0)              # pilot, +y wall
    s = s - c * Pos(0, FLAT_Y + 5, CLAMP_Z - 4.0) * Box(9.0, 10, 8.0, align=MIN)            # spot face
    sol = s.solids()
    assert len(sol) == 1, ("tower mods split the spine", [round(x.volume) for x in sol])
    return sol[0]


def sleeve_slot_z():
    """Sleeve-frame z range of the screw slot over the shim range (sleeve bottom on the T bar)."""
    lo = CLAMP_Z - (T.FLOOR_Z + SHIM_RANGE[1] + T.T_THICK)
    hi = CLAMP_Z - (T.FLOOR_Z + SHIM_RANGE[0] + T.T_THICK)
    return lo - SLEEVE_SLOT_W / 2 - 0.3, hi + SLEEVE_SLOT_W / 2 + 0.3


def sleeve():
    """Sleeve, assembly orientation (bottom on the T bar at z = 0), prints upright. Pin bore: V on -y, open to +y;
    screw slot through the +y wall."""
    L = T.PEG_L
    p = Cylinder(SLEEVE_R, L, align=MIN)
    w, top = PIN_SLOT_W / 2, PIN_SLOT_TOP
    bore = Polygon((0, -PIN_V_APEX), (w, w - PIN_V_APEX), (w, top), (-w, top), (-w, w - PIN_V_APEX), align=None)
    p -= Pos(0, 0, -1) * extrude(bore, L + 2)
    z0, z1 = sleeve_slot_z()
    p -= Pos(0, 3.15, z0) * Box(SLEEVE_SLOT_W, 5.7, z1 - z0, align=MIN)      # flat top: a 3.4 mm bridge. It reaches
                                                                              # down to the pin (y 0.3): the M3 is
                                                                              # wider than the pin bore
    sol = p.solids()
    assert len(sol) == 1
    return sol[0]


def clamp_screw():
    """M3x12 ISO 7380 button head, tower frame (pointing -y), turned in until its tip is on the pin (the head then
    stands SCREW_L - (FLAT_Y - PIN_R) off the spot face: that much thread is left to tighten)."""
    tip = PIN_R + 0.02
    s = Pos(0, tip + SCREW_L / 2, CLAMP_Z) * Rot(X=90) * Cylinder(1.5, SCREW_L)
    s += Pos(0, tip + SCREW_L + 0.825, CLAMP_Z) * Rot(X=90) * Cylinder(2.85, 1.65)
    return s


# ---- damper -----------------------------------------------------------------------------------------------------
HUB_Z = T.Z_ARM                   # hub underside (79.5)
CUP_FLOOR = (T.TOWER_H, T.TOWER_H + 1.2)
CUP_IN = (SLEEVE_R + BORE_CLR, 7.6)        # inner tube around the sleeve: r 5.35-7.6 (centres the cup on the sleeve)
CUP_OUT = (10.7, 12.3)                     # outer wall
CUP_IN_TOP, CUP_OUT_TOP = 76.8, 77.3
SKIRT_R = (8.6, 9.7)                       # 1.0 mm gap to each wall
WASHER_R = (5.8, 9.7)
WASHER_T = 1.2
SKIRT_CLR = 2.5                            # skirt bottom above the cup floor (the arm may sit up to ~2 mm lower)
DOWEL_XY = ((0.0, 7.2), (0.0, -7.2))       # 1.75 filament dowels: hub underside to the washer
DOWEL_HOLE, DOWEL_DEPTH = 2.05, 5.0        # as the headshell's grooves: prints ~1.8, snug on 1.75


def cup():
    """Damper cup, turntable frame (centred on the tower axis at the origin; place with Pos(ARM_D, 0, 0)). Prints
    as it sits, floor on the bed."""
    z0, z1 = CUP_FLOOR
    c = Pos(0, 0, z0) * Cylinder(CUP_OUT[1], z1 - z0, align=MIN)
    c += Pos(0, 0, z0) * (Cylinder(CUP_IN[1], CUP_IN_TOP - z0, align=MIN) - Cylinder(CUP_IN[0], 99, align=MIN))
    c += Pos(0, 0, z0) * (Cylinder(CUP_OUT[1], CUP_OUT_TOP - z0, align=MIN) - Cylinder(CUP_OUT[0], 99, align=MIN))
    c -= Pos(0, 0, z0 - 1) * Cylinder(CUP_IN[0], 99, align=MIN)
    # 0.8 mm fillets where both walls meet the floor (they face up as printed): wall roots, and no sharp corner
    # for the liquid to creep from
    c, _ = T.fillet_inside(c.solids()[0], 0.8, up=(0, 0, 1), keep=lambda p: abs(p.Z - z1) < 0.1)
    return c


def skirt_bottom_z():
    return CUP_FLOOR[1] + SKIRT_CLR


def skirt():
    """Hub skirt, arm frame (hub underside z = 0, the part below it). Prints washer down (flip it)."""
    zb = skirt_bottom_z() - HUB_Z
    s = Pos(0, 0, -WASHER_T) * (Cylinder(WASHER_R[1], WASHER_T, align=MIN) - Cylinder(WASHER_R[0], 9, align=MIN))
    s += Pos(0, 0, zb) * (Cylinder(SKIRT_R[1], -zb, align=MIN) - Cylinder(SKIRT_R[0], 99, align=MIN))
    for x, y in DOWEL_XY:
        s -= Pos(x, y, -WASHER_T - 1) * Cylinder(DOWEL_HOLE / 2, WASHER_T + 2, align=MIN)
    sol = s.solids()
    assert len(sol) == 1
    # 0.6 mm at the ring's root under the washer (prints washer down, so it faces up)
    out, _ = T.fillet_inside(sol[0], 0.6, up=(0, 0, -1),
                             keep=lambda p: abs(p.Z + WASHER_T) < 0.1 and abs(math.hypot(p.X, p.Y) - SKIRT_R[0]) < 0.1)
    return out


def dowels():
    from build123d import Compound
    return Compound(children=[Pos(x, y, -WASHER_T) * Cylinder(0.875, WASHER_T + DOWEL_DEPTH - 0.5, align=MIN)
                              for x, y in DOWEL_XY])


def arm_holes(a):
    """The new arm with the skirt's dowel holes in the hub underside (blind, printed from the bed up)."""
    for x, y in DOWEL_XY:
        a = a - Pos(x, y, -0.5) * Cylinder(DOWEL_HOLE / 2, DOWEL_DEPTH + 0.5, align=MIN)
    return a


# ---- acorn-nut carriage -----------------------------------------------------------------------------------------
NUT_AF, NUT_LEN, NUT_G = 5.5, 7.0, 1.0     # M3 acorn nut (DIN 1587); mass assumed, weigh yours
NUT_HOLE_AF = 5.8                          # horizontal hex, vertex up: prints ~5.5, a sliding friction fit
CAR_L, CAR_TOP = 22.0, 30.0                # carriage frame as turntable.weight(): channel floor at 19.5
CAR_BOT = -13.4                            # bottom (arm frame z -33.2): clears the wire anchor fin (top 42.3) by
                                           # 3.7 mm with the stylus lifted 10 mm at the inner groove
NUT_COLS = (-7.2, 0.0, 7.2)
NUT_WALL = 1.2


def nut_rows():
    h = NUT_HOLE_AF / math.cos(math.radians(30))          # vertex to vertex, 6.70
    r1 = CAR_BOT + NUT_WALL + h / 2
    return (r1, r1 + h + 0.9), h


def carriage():
    """Counterweight carriage, same frame as turntable.weight() (channel floor at z 19.5, placed with
    Pos(x, 0, -19.8)): the stub channel and friction tongue on top, hex holes along y for 12 M3 acorn nuts below.
    Prints roof down (flip it): the hex holes are vertex-up both ways; the channel and the window are bridges."""
    L = CAR_L
    W = 2 * (T.WEIGHT_Y + T.BEAM / 2 + 0.3 + 2.0)
    yc = T.WEIGHT_Y
    w = Pos(0, yc, CAR_BOT) * Box(L, W, CAR_TOP - CAR_BOT, align=MIN)
    z0 = 30 - 2 - T.BEAM - 0.5                                          # channel floor (19.5)
    w -= Pos(0, 0, z0) * Box(L + 1, T.BEAM + 0.6, T.BEAM + 0.6, align=MIN)
    (r1, r2), h = nut_rows()
    from build123d import RegularPolygon
    hexa = Rot(X=-90) * extrude(Rot(Z=90) * RegularPolygon(NUT_HOLE_AF / math.sqrt(3), 6), W + 2)   # along +y, vertex up
    for x in NUT_COLS:
        for z in (r1, r2):
            w -= Pos(x, yc - W / 2 - 1, z) * hexa
    win_z0 = r2 + h / 2 + 1.5
    w -= Pos(0, yc, win_z0) * Box(L - 6, W + 2, z0 - 2.0 - win_z0, align=MIN)      # lightening window (high mass is
                                                                                    # the wrong place for it)
    # friction tongue on the channel's -y wall, as turntable.weight()
    yw = -(T.BEAM + 0.6) / 2
    tx, t_top = T.WT_TONGUE_W / 2, z0 + T.WT_TONGUE_L
    bx = lambda x0, x1, y0, y1, za, zb: Pos((x0 + x1) / 2, (y0 + y1) / 2, za) * Box(x1 - x0, y1 - y0, zb - za, align=MIN)
    yo = yc - W / 2
    for xa in (-tx - 0.6, tx):
        w -= bx(xa, xa + 0.6, yo - 1, yw + 0.01, z0 - 0.5, t_top)
    w -= bx(-tx, tx, yo - 1, yw + 0.01, z0 - 0.5, z0 + 0.1)
    w -= bx(-tx, tx, yo - 1, yw - T.WT_TONGUE_T, z0 - 0.5, t_top)
    b = T.WT_BUMP
    prof = [(tx - 0.5, yw), (tx - 0.5 - b, yw + b), (-tx + 0.5 + b, yw + b), (-tx + 0.5, yw)]
    w += Pos(0, 0, z0 + 0.1) * extrude(Polygon(*prof, align=None), 3.0)
    sol = w.solids()
    assert len(sol) == 1, [round(x.volume) for x in sol]
    return sol[0]


def nut_positions(n, shift=0.0):
    """Arm-frame-free carriage-frame centres of n acorn nuts (fill order: low row first, both ends of each hole),
    each pushed `shift` mm toward +y from flush with the faces."""
    W = 2 * (T.WEIGHT_Y + T.BEAM / 2 + 0.3 + 2.0)
    (r1, r2), _ = nut_rows()
    slots = []
    for z in (r1, r2):
        for x in NUT_COLS:
            for side in (-1, 1):
                y = T.WEIGHT_Y + side * (W / 2 - NUT_LEN / 2)
                slots.append(np.array([x, y + shift, z]))
    return slots[:n]


def install():
    """Point turntable.py at the steady parts (after headshell.install(), which spine.py runs first)."""
    if getattr(T, "_steady", False):
        return
    T._steady = True
    _arm = T.arm
    T.arm = lambda: arm_holes(_arm())
    T.peg = sleeve
    T.weight = carriage


# ---- checks -------------------------------------------------------------------------------------------------------
def balance(force_g=5.0, carriage_x=-34.0, shift=0.0, verbose=False):
    """Nuts for a tracking force with the carriage at carriage_x, the CG depth under the pivot-stylus line and the
    static roll (as balance.py, with nuts in their holes instead of coins in a cup)."""
    import balance as B
    import headshell as HS
    rho, fill = B.RHO, B.FILL
    ma, ca = B.mass_cg(T.arm())
    sgn = T.headshell_sign()
    shell = Pos(T.ARM_L, 0, 0) * Rot(Z=180 + sgn * T.ARM_BETA)
    cc = B.mass_cg(shell * T.cartridge(), 1)[1]
    mh, ch = B.mass_cg(shell * T.cart_mount())
    leads = np.array([70.0, 0.0, T.BEAM / 2])
    ms, cs = B.mass_cg(skirt(), 0.9)
    place = np.array([carriage_x, 0, -(30 - 2 - T.BEAM - 0.5) - 0.3])
    mw, cw = B.mass_cg(carriage())
    cw = cw + place
    fixed = [(ma, ca), (B.M_CART, cc), (B.M_LEADS, leads), (mh + 0.4, ch), (ms, cs), (mw, cw)]
    moment = sum(m * c[0] for m, c in fixed)
    need = (moment - force_g * T.ARM_L) / (-carriage_x)              # g of nuts at the carriage's x
    n = max(0, min(12, round(need / NUT_G)))
    nuts = [(NUT_G, p + place) for p in nut_positions(n, shift)]
    parts = fixed + nuts
    m = sum(p[0] for p in parts)
    g = sum(p[0] * p[1] for p in parts) / m
    f_real = sum(p[0] * p[1][0] for p in parts) / T.ARM_L          # moment about the pivot / L: force at the stylus
    tip_z = T.CONE_H + T.PIVOT_RISE - 0.05
    a, b = np.array([0, 0, tip_z]), np.array([T.ARM_L, 0, -T.CART_H])
    u = (b - a) / np.linalg.norm(b - a)
    v = g - a - ((g - a) @ u) * u
    # computed in the arm's design frame; arm_side() mirrors the real arm, so the sideways offset and roll flip
    return dict(need=need, nuts=n, force=float(f_real), mass=float(m), depth=float(-v[2]), side=float(T.ARM_SIDE * v[1]),
                roll=math.degrees(math.atan2(T.ARM_SIDE * v[1], -v[2])), carriage_g=mw)


def damper_estimate(mu=0.5, fill_above_floor=5.0, f_roll=2.8, depth=None, mass=None):
    """Damping ratio of the roll mode from short squeeze-film theory (both gaps; order-of-magnitude only)."""
    L = max(0.0, CUP_FLOOR[1] + fill_above_floor - skirt_bottom_z()) * 1e-3
    c_lat = 0.0
    for R, gap in ((SKIRT_R[0] - 0.5 * (SKIRT_R[0] - CUP_IN[1]), SKIRT_R[0] - CUP_IN[1]),
                   (SKIRT_R[1] + 0.5 * (CUP_OUT[0] - SKIRT_R[1]), CUP_OUT[0] - SKIRT_R[1])):
        c_lat += math.pi * mu * (R * 1e-3) * L ** 3 / (2 * (gap * 1e-3) ** 3)
    tip = T.Z_ARM + T.CONE_H + T.PIVOT_RISE
    lever = (tip - (skirt_bottom_z() + L * 1e3 / 2)) * 1e-3
    c = c_lat * lever ** 2
    k = mass / 1000 * 9.81 * depth / 1000
    w = 2 * math.pi * f_roll
    return c / (2 * k / w), L * 1e3


def checks():
    from motion import dist
    bad = []

    def say(msg, ok=True):
        print(("  " if ok else "  PROBLEM: ") + msg)
        if not ok:
            bad.append(msg)

    print("rigid pivot")
    sl = sleeve()
    tower_c = Pos(T.ARM_D, 0, 0)
    import spine as S
    sp = S.spine()
    zb = T.peg_bottom_z()
    sl_w = Pos(T.ARM_D, 0, zb) * sl
    pin = Pos(T.ARM_D, 0, T.TIP_Z + T.PIVOT_RISE - T.PIN_LEN) * T.tpin()
    d_sv = dist(sl_w, sp)
    say(f"sleeve seated in the tower V: gap {d_sv:.3f} mm (V flats tangent to the 10.0 sleeve)", d_sv < 0.02)
    d_pv = dist(pin, sl_w)
    say(f"pin in the sleeve's V: gap {d_pv:.3f} mm", d_pv < 0.02)
    scr = tower_c * clamp_screw()
    tip_y = FLAT_Y - SCREW_L
    reach = PIN_R - tip_y
    say(f"M3x{SCREW_L:.0f} clamp at z {CLAMP_Z}: tip reaches {reach:.1f} mm past the pin's near side (pilot "
        f"{2 * PILOT_R} mm through the +y wall, {T.TOWER_OD / 2 - FLAT_Y + 0.0:.1f} mm spot face)", reach > 0.8)
    z0, z1 = sleeve_slot_z()
    for shim in SHIM_RANGE:
        zs = CLAMP_Z - (T.FLOOR_Z + shim + T.T_THICK)
        say(f"  shims {shim:.0f} mm: screw at sleeve z {zs:.1f}, slot {z0:.1f}-{z1:.1f}",
            z0 + 1.7 <= zs <= z1 - 1.7)
    ov = dist(scr, sl_w)
    say(f"screw on the pin clears the sleeve's slot by {ov:.2f} mm",
        ov > 0.1)
    say(f"tower wall behind the V apex {T.TOWER_OD / 2 - V_APEX:.1f} mm; the pilot runs {T.TOWER_OD / 2 - SLEEVE_R - BORE_CLR:.1f} mm "
        f"through the +y wall ({FLAT_Y - SLEEVE_R - BORE_CLR:.1f} mm of thread after the spot face)",
        T.TOWER_OD / 2 - V_APEX > 3 and FLAT_Y - SLEEVE_R - BORE_CLR > 4.5)

    print("damper")
    cp = tower_c * cup()
    ov = dist(cp, sp) ; say(f"cup on the tower top: gap {ov:.3f} mm", ov < 0.02)
    say(f"cup centred by the sleeve: inner tube {2 * CUP_IN[0]:.1f} on the {T.PEG_D:.1f} sleeve (prints ~0.3 under)")
    A = Pos(T.ARM_D, 0, T.Z_ARM) * Rot(Z=T.arm_angle(65))
    sk = A * T.arm_side(skirt())
    d1, d2 = dist(sk, cp), dist(sk, sl_w)
    say(f"skirt in the cup: {d1:.2f} mm to the walls (gaps {SKIRT_R[0] - CUP_IN[1]:.1f} / {CUP_OUT[0] - SKIRT_R[1]:.1f}), "
        f"{skirt_bottom_z() - CUP_FLOOR[1]:.1f} above the floor; washer {d2:.2f} mm over the sleeve top",
        d1 > 0.9 and d2 > 0.6)
    from headshell import common
    ov = common(sk, A * T.arm_side(T.arm())) + common(A * T.arm_side(dowels()), A * T.arm_side(T.arm())) * 0
    say(f"skirt | arm overlap {ov:.3f} mm3 (washer flat on the hub underside, two dowels)", ov < 0.01)
    d3 = dist(A * T.arm_side(T.arm()), cp)
    say(f"cup top to the arm (hub, stub, carriage): {d3:.2f} mm", d3 > 1.5)
    for tilt in (2.5, -2.5):
        Ap = Pos(T.ARM_D, 0, T.Z_ARM + T.CONE_H + T.PIVOT_RISE) * Rot(Z=T.arm_angle(65)) * Rot(Y=tilt) * \
            Pos(0, 0, -(T.CONE_H + T.PIVOT_RISE))
        d = min(dist(Ap * T.arm_side(skirt()), cp), dist(Ap * T.arm_side(skirt()), sl_w))
        say(f"arm pitched {tilt:+.1f} deg (stylus {'lifted' if tilt > 0 else 'dropped'} ~{abs(tilt) * 2.36:.0f} mm): skirt "
            f"clears by {d:.2f} mm", d > 0.15)
    stop = math.degrees(math.asin((CUP_OUT[0] - SKIRT_R[1]) / (T.Z_ARM + T.CONE_H + T.PIVOT_RISE - skirt_bottom_z())))
    say(f"pitch stop: the skirt meets the cup wall at about +-{stop:.1f} deg (stylus +-{stop * 2.36:.0f} mm); lift the "
        f"arm off by the finger lift, not past that")

    print("carriage and balance (5 g at the stylus)")
    for f in (4.0, 5.0, 6.0):
        s = balance(f)
        print(f"  {f:.0f} g: {s['need']:.1f} g of nuts -> {s['nuts']} acorn nuts ({NUT_G} g each), arm {s['mass']:.1f} g, "
              f"CG {s['depth']:.1f} mm under the pivot-stylus line, {s['side']:+.2f} sideways -> roll {s['roll']:+.1f} deg")
    s0 = balance(5.0)
    for sh in (-2.0, 2.0):
        s = balance(5.0, shift=sh)
        print(f"    nuts pushed {sh:+.0f} mm along their holes: roll {s['roll']:+.1f} deg")
    say(f"carriage {s0['carriage_g']:.1f} g of PLA; 12 hex holes for M3 acorn nuts (AF {NUT_AF}, drawn {NUT_HOLE_AF})",
        s0["nuts"] <= 12)
    for mu, name in ((0.06, "cooking oil"), (0.5, "dish soap"), (5.0, "honey")):
        for fill in (4.0, 6.0):
            z, L = damper_estimate(mu, fill, depth=s0["depth"], mass=s0["mass"])
            print(f"  damper, {name} ({mu} Pa s) {fill:.0f} mm deep ({L:.1f} mm on the skirt): roll damping ratio ~{z:.2f}")
    return bad


if __name__ == "__main__":
    import spine  # noqa: F401  (installs the headshell and these parts)
    b = checks()
    print(f"{len(b)} problem(s)")
