"""
rim_clamp.py - a twist-lock clamp ring that holds the printed 7 in disc's edge flat on the open platter (Nov 24 2026).

    python rim_clamp.py       # checks: fit on the platter, lock range, clearance to the cartridge at the lead-in

Why: printed PLA discs warp edges-up (about 1.0 mm thick at the edge, lifting under 1 mm, Nov 24) and had to be
taped down. The ring presses the disc's ungrooved edge band (r 86 to the 88.9 edge; the groove ends at 84) onto
the platter rim.

Height: with the stylus on the first groove (r 84) the cartridge body is only 1.9 mm above the record from r 88 to
90, and the headshell 3-5.5 mm up as far out as r 96. So the ring stands 1.0 mm above the record at most: a 1 mm
flange over the disc edge and across to a skirt round the outside of the platter rim.

Lock: six inward lugs at the foot of the skirt drop through notches in the platter rim; turning the ring clockwise
(seen from above) slides them under ramped ledges cut into the rim's outer face, which pull the ring down 1.6 mm
over 25 deg (a 2 deg ramp: friction holds it, no detent). Turn it until snug: the ramp takes any disc 0.4-2.0 mm
thick. The bearing faces are 45 deg (lug top, ledge underside) so both parts print without supports: the platter
top face down, the ring flange down.

Platter frame: as turntable.platter(), z = 0 on the horn, top at HUB_H = 14; ring frame: flange top at z = 0.
"""
import math

from build123d import Align, Box, Cone, Cylinder, Plane, Polygon, Pos, Rot, extrude

import turntable as T

MIN = (Align.CENTER, Align.CENTER, Align.MIN)
DISC_T = 1.0                     # printed disc at the edge (Nov 24)
FLANGE_T = 1.0
R_IN = 86.0                      # flange inner edge: the groove ends at 84, and cueing by hand can land at 85
SKIRT_IN, SKIRT_OUT = T.PLAT_R + 0.4, T.PLAT_R + 2.45      # 90.4 / 92.45
N_LUG = 6
LUG_R_IN = 89.3                  # lug tip
LUG_W = 6.0                      # along the rim
LUG_FLAT, LUG_H = 0.5, 1.65      # lug: flat 0.5 at the tip, 45 deg top face up to the skirt
LUG_ZB = -6.5                    # lug underside, ring frame
NOTCH_R = 89.1                   # notch and recess inner radius in the rim (the disc edge is 88.9: still supported)
NOTCH_W = LUG_W + 1.0
TRAVEL = 25.0                    # deg of twist from the notch to the end of the ramp
LEDGE_HI, LEDGE_LO = 11.2, 9.4   # platter z of the 45 deg ledge at the lug tip radius: entry, end of travel
STEP = 1.0                       # ramp built from 1 deg steps (0.07 mm each)
RING_FILLET = 0.6                # skirt-to-flange root


def lug_angles():
    return [360.0 * k / N_LUG for k in range(N_LUG)]          # spokes are at 22.5 + 45 k: lugs clear them in angle


def ledge_at(frac):
    return LEDGE_HI + (LEDGE_LO - LEDGE_HI) * frac


def ring_top_z(disc_t):
    """Ring top in the platter frame once snug on a disc of thickness disc_t."""
    return T.HUB_H + disc_t + FLANGE_T


def disc_range():
    """Disc thicknesses the ramp can clamp: the lug's 45 deg face meets the ledge where (platter frame, at the lug
    tip radius) ledge z = ring top + LUG_ZB + LUG_FLAT."""
    t = lambda ledge: ledge - LUG_ZB - LUG_FLAT - T.HUB_H - FLANGE_T
    return t(LEDGE_LO), t(LEDGE_HI)


def ring():
    """Ring frame: flange top at z = 0. Print flange down (flip it)."""
    r = Pos(0, 0, -FLANGE_T) * (Cylinder(SKIRT_OUT, FLANGE_T, align=MIN) - Cylinder(R_IN, 9, align=MIN))
    r += Pos(0, 0, LUG_ZB) * (Cylinder(SKIRT_OUT, -LUG_ZB - FLANGE_T + 0.01, align=MIN) - Cylinder(SKIRT_IN, 99, align=MIN))
    prof = Polygon((SKIRT_IN + 0.3, LUG_ZB), (LUG_R_IN, LUG_ZB), (LUG_R_IN, LUG_ZB + LUG_FLAT),
                   (SKIRT_IN + 0.3, LUG_ZB + LUG_FLAT + (SKIRT_IN + 0.3 - LUG_R_IN)), align=None)
    lug = Pos(0, -LUG_W / 2, 0) * extrude(Plane.XZ * prof, -LUG_W)       # profile in (x, z), LUG_W along y
    for a in lug_angles():
        r += Rot(Z=a) * lug
    sol = r.solids()
    assert len(sol) == 1, [round(x.volume) for x in sol]
    # 0.6 mm where the skirt meets the flange: the clamp load runs from the flange through this corner to the lugs.
    # Faces up as printed (flange down).
    out, _ = T.fillet_inside(sol[0], RING_FILLET, up=(0, 0, -1),
                             keep=lambda p: abs(p.Z + FLANGE_T) < 0.1 and abs(math.hypot(p.X, p.Y) - SKIRT_IN) < 0.1)
    return out


def platter_cuts():
    """Notches and ramped recesses in the rim's outer face, platter frame. Lock = ring turned clockwise from above,
    so seen from the platter each lug travels clockwise (negative angle) from its notch."""
    out = None
    z_lo = T.HUB_H - T.PLATE_T - T.RIB_H - 1.0                   # below the rim
    for a in lug_angles():
        notch = Pos(0, 0, z_lo) * Box(20, NOTCH_W, 30, align=MIN)
        notch = Rot(Z=a) * Pos(NOTCH_R + 10, 0, 0) * notch
        cut = notch
        half = math.degrees((NOTCH_W / 2) / T.PLAT_R)
        n = int(round((TRAVEL + math.degrees(LUG_W / T.PLAT_R)) / STEP))
        for i in range(n + 1):
            frac = min(1.0, i * STEP / TRAVEL)
            L = ledge_at(frac)
            # region below the 45 deg ledge, which passes through (LUG_R_IN, L) rising outward
            pr = Polygon((NOTCH_R, z_lo), (T.PLAT_R + 2, z_lo), (T.PLAT_R + 2, L + (T.PLAT_R + 2 - LUG_R_IN)),
                         (NOTCH_R, L - (LUG_R_IN - NOTCH_R)), align=None)
            w = math.radians(STEP * 1.3) * T.PLAT_R
            seg = Pos(0, -w / 2, 0) * extrude(Plane.XZ * pr, -w)
            ang = a - half - STEP * (i + 0.5) + 0.3
            cut += Rot(Z=ang) * seg
        out = cut if out is None else out + cut
    return out


def add_to_platter(p):
    """Open platter (platter frame) -> with the notches and ramped recesses."""
    p = p - platter_cuts()
    sol = p.solids()
    assert len(sol) == 1, [round(x.volume) for x in sol]
    return sol[0]


# ---- centre weight (Nov 24): a puck over the spindle on the label area, loaded with M3 acorn nuts -----------
PUCK_R, PUCK_H = 25.0, 8.0       # at the lead-out (stylus r 44) the arm comes in to r 27.7 at 13.4 mm above the record,
                                 # the headshell to r 32 at 5.3 mm: the puck clears both
PUCK_BORE = 7.8                  # over the 6.85-7.25 spindle (prints ~7.5); the spindle stands 4 mm above a 1 mm disc
PUCK_NUT_AF, PUCK_NUT_DEPTH = 5.8, 6.0   # vertical hex pockets, prints ~5.5: a friction fit; the 7 mm nut stands 1 mm
                                         # proud, dome up, so it pulls out with fingers
PUCK_RINGS = ((10.5, 6), (18.5, 12))     # (radius, pockets): 18 nuts


def centre_weight():
    """Record weight, its own frame: underside on the disc at z = 0. Prints as it sits. Fill the pockets
    symmetrically so the platter stays balanced."""
    from build123d import RegularPolygon
    w = Cylinder(PUCK_R, PUCK_H, align=MIN) - Pos(0, 0, -1) * Cylinder(PUCK_BORE / 2, PUCK_H + 2, align=MIN)
    pocket = extrude(RegularPolygon(PUCK_NUT_AF / math.sqrt(3), 6), PUCK_NUT_DEPTH + 1)
    for r, n in PUCK_RINGS:
        for k in range(n):
            a = 360.0 * (k + 0.5 * (r < 15)) / n
            w -= Rot(Z=a) * Pos(r, 0, PUCK_H - PUCK_NUT_DEPTH) * Rot(Z=30) * pocket
    w -= Pos(0, 0, PUCK_H - 0.6) * Cone(PUCK_BORE / 2, PUCK_BORE / 2 + 0.8, 0.8, align=MIN)   # lead-in on top
    sol = w.solids()
    assert len(sol) == 1
    return sol[0]


def snug_turn(disc_t):
    """Clockwise turn from the notches at which the lug's leading edge meets the ledge on a disc_t disc."""
    need = ring_top_z(disc_t) + LUG_ZB + LUG_FLAT
    frac = (need - LEDGE_HI) / (LEDGE_LO - LEDGE_HI)
    return frac * TRAVEL + math.degrees((NOTCH_W / 2 - LUG_W / 2) / T.PLAT_R)


def in_platter_frame(disc_t=DISC_T):
    """The ring placed on the platter, snug on disc_t."""
    return Pos(0, 0, ring_top_z(disc_t)) * Rot(Z=-snug_turn(disc_t)) * ring()


def checks(platter=None):
    from motion import dist
    import spine as S
    import spine_assembly as SA
    bad = []

    def say(msg, ok=True):
        print(("  " if ok else "  PROBLEM: ") + msg)
        if not ok:
            bad.append(msg)

    lo, hi = disc_range()
    say(f"ramp: {LEDGE_HI - LEDGE_LO:.1f} mm over {TRAVEL:.0f} deg at r {LUG_R_IN} "
        f"({math.degrees(math.atan((LEDGE_HI - LEDGE_LO) / math.radians(TRAVEL) / LUG_R_IN)):.1f} deg slope); clamps a disc "
        f"{lo:.1f}-{hi:.1f} mm thick", lo < 0.7 and hi > 1.6)
    p = platter if platter is not None else S.open_platter()
    from headshell import common
    turn = snug_turn(DISC_T)
    say(f"{DISC_T:.1f} mm disc: snug after turning {turn:.0f} deg clockwise from the notches")
    # interference: drop the ring straight down through the notches, then turn it to just short of snug
    for t_disc in (0.6, DISC_T, 1.6):
        for deg in (0.0, snug_turn(t_disc) - 1.0):
            rg = Pos(0, 0, ring_top_z(t_disc)) * Rot(Z=-deg) * ring()
            ov = common(rg, p)
            say(f"ring on a {t_disc:.1f} mm disc, turned {deg:4.1f} deg: overlap with the platter {ov:.3f} mm3", ov < 0.01)
    rg = Pos(0, 0, ring_top_z(DISC_T)) * Rot(Z=-(turn - 1)) * ring()
    rec = Pos(0, 0, T.HUB_H) * (Cylinder(T.R_REC, DISC_T, align=MIN) - Cylinder(3.62, 9, align=MIN))
    d = dist(rg, rec)
    say(f"flange on the disc edge band r {R_IN}-{T.R_REC:.1f}: gap {d:.3f} mm", d < 0.02)
    # the arm at the lead-in: ring in the turntable frame on the assembly's 1.2 mm record
    t_asm = T.Z_REC - T.Z_HORN - T.HUB_H
    rg_w = Pos(0, 0, T.Z_HORN + ring_top_z(t_asm)) * Rot(Z=-snug_turn(t_asm)) * ring()
    for r in (84.0, 85.0):
        a = SA.assembly(r_stylus=r)
        dm = min(dist(a[k][0], rg_w) for k in ("cartridge (CKANDAY)", "headshell", "arm"))
        say(f"stylus at r {r:.0f}: cartridge / headshell / arm to the ring {dm:.2f} mm", dm > 0.6)
    return bad


if __name__ == "__main__":
    import spine  # noqa: F401
    b = checks()
    print(f"{len(b)} problem(s)")
