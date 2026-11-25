"""
OpenVinyl weekend turntable: direct drive on a Feetech STS3215, unipivot tonearm.

    python turntable.py            # writes stl/*.stl (print orientation) and turntable_preview.png

Parts
  platter   180 mm, bolts to the stock 20 mm servo horn through four radial slots
            (M3, any hole circle 10-18 mm), printed top face down
  spindle   7.0 mm pin for the 7.24 mm hole of the printed disc, pressed into the platter
  base      servo pocket + beam + pivot tower + arm rest, one piece (about 250 mm long)
  peg       sleeve over the pivot pin: a whole 1.1 mm T-pin stands point-up with its T bar in a
            slot at the bottom of the tower bore, on printed shims; the sleeve slides over it
  shims     1, 2 and 4 mm bars under the T bar set the arm height (3 mm nominal)
  arm       hub with a pivot cone, 150 mm effective length, headshell with 1/2 in
            mount slots at a 25.1 deg offset, rear stub for the counterweight
  weight    counterweight carriage: slides on the stub, cup for coins or nuts

Every dimension that comes from the servo, the horn or the cartridge is a constant
below. Servo values were measured Nov 20; cartridge height is still an estimate (the shims absorb it).
"""

from pathlib import Path
import math

import numpy as np
from build123d import Box, Cylinder, Cone, Pos, Rot, Align, export_stl

OUT = Path(__file__).parent / "stl"

# ---- record and tonearm geometry ----------------------------
R_REC = 177.8 / 2
GROOVE_R = (46.0, 84.0)
ARM_L = 150.0             # pivot to stylus
ARM_D = 136.51            # pivot to spindle (solved for min max tracking error)
ARM_BETA = 25.08          # headshell offset angle, deg; max error 0.60 deg, nulls r 50.2 / 77.0
# Which side of the pivot-spindle line the stylus swings on (Nov 25 2026). The platter turns clockwise seen from
# above (the only way the spiral plays forwards), and the groove must run under the stylus from the cartridge's
# pins toward its tip. With the stylus on +y (the original drawing) the only offset that lines the cartridge up
# with the groove drags the needle backwards into its mount; the other offset is 57 deg off (what the first
# prototype ran, thumping on 9 of 13 landings). On -y the aligned offset also drags the right way. The arm, its
# headshell pad, tie tab, drop tab and the counterweight carriage are built in the original (+y) design frame and
# mirrored by arm_side(); stylus_xy() and anchor_xy() put them on this side in the turntable frame.
ARM_SIDE = -1


def arm_side(shape):
    """A part built in the arm's design frame, mirrored to the side the stylus actually swings on."""
    if ARM_SIDE > 0:
        return shape
    from build123d import Plane, mirror
    return mirror(shape, about=Plane.XZ)


def groove_drag_ok(r=65.0):
    """(offset from the groove in deg, True if the groove runs pins -> stylus) at radius r, clockwise platter."""
    S = stylus_xy(r)
    P = np.array([ARM_D, 0.0])
    rear = (P - S) / np.linalg.norm(P - S)
    a = math.radians(ARM_SIDE * headshell_sign() * ARM_BETA)
    c = np.array([rear[0] * math.cos(a) - rear[1] * math.sin(a), rear[0] * math.sin(a) + rear[1] * math.cos(a)])
    v = np.array([S[1], -S[0]]) / np.linalg.norm(S)
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(c, v)))))), bool(np.dot(v, c) < 0)

# ---- servo (Feetech STS3215) -----------------------------------------------------------
SERVO_L, SERVO_W, SERVO_H = 45.2, 24.7, 35.0   # body, datasheet
SHAFT_FROM_END = 9.5      # measured Nov 20: horn centre to the near short end of the body
HORN_OD = 20.0            # measured
# Printed holes come out small (Nov 22: the sleeve, spindle socket and pivot counterbore were all tight): a 0.4 mm
# nozzle shrinks small vertical holes 0.2-0.4 mm, and horizontal ones sag at the top. Free fits below are drawn
# about 0.3 mm over (0.4 for horizontal), self-tap pilots 0.2 over, so they print at the size they need.
HORN_SCREW = 3.5          # M3 clearance (3.2 printed tight)
FIT = 0.4                 # pocket clearance per side

# ---- platter ----------------------------------------------------------------------------
PLAT_R = 90.0
PLATE_T = 3.0
RIB_H = 5.0
HUB_R = 16.0
HUB_H = 14.0              # horn top to platter top; ribs start 6 mm above the horn to clear the body top (the horn sits in a recess)
SPINDLE_D = 7.0
SOCKET_D = 5.4            # spindle socket, a through-hole (the spindle's pin is 5.0)
# Nov 22, first plays: the spindle sat loose and the record ran off centre. Cause: the four counterbore slots
# (CB_D wide, centres from r 5) reach in to r 1.4, inside the 5.4 socket, so above SEAT_Z the socket has almost no
# wall and the round pin had nothing to grip. The spindle now has a key that fills the socket and the four slots
# (KEY_CLR smaller all round) from KEY_Z0 up to the platter top: the slot walls locate it. The part above the
# platter comes as a fit set for the printed disc hole (drawn 7.24, prints 6.9-7.2): pick the snug one.
SPINDLE_SET = (6.85, 6.95, 7.05, 7.15, 7.25)   # mm; 1 to 5 dimples on the top, smallest first
SPINDLE_UP = 4.2          # straight part above the platter (was 3.2): still holds a 1.2 mm disc if the key sits 1 mm low
CB_D = 7.2                # M3 pan-head counterbore (servo-bag screws have a ~6.3 washer face)
SEAT_Z = 3.0              # screw head seat height above the platter underside
KEY_Z0 = SEAT_Z + 3.0     # spindle key bottom: it rests on the four horn screw heads (M3 pan heads stand 2.0-2.5,
                          # so it sits flush or up to 1 mm low; nothing else holds its height)
KEY_CLR = 0.15            # key to socket and slot walls, per side


def slot(x0, x1, width, height, z0=0.0):
    """Stadium-shaped slot along x, from x0 to x1 (centres), extruded in z."""
    r = width / 2
    s = Pos((x0 + x1) / 2, 0, z0) * Box(x1 - x0, width, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
    s += Pos(x0, 0, z0) * Cylinder(r, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
    s += Pos(x1, 0, z0) * Cylinder(r, height, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return s


def _edge_faces(shape):
    """{edge: [faces]} from OCP, keyed by build123d Edge (equality is IsSame)."""
    from OCP.TopExp import TopExp
    from OCP.TopTools import TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
    from build123d import Edge, Face
    m = TopTools_IndexedDataMapOfShapeListOfShape()
    TopExp.MapShapesAndAncestors_s(shape.wrapped, TopAbs_EDGE, TopAbs_FACE, m)
    out = []
    for i in range(1, m.Extent() + 1):
        out.append((Edge(m.FindKey(i)), [Face(f) for f in m.FindFromIndex(i)]))
    return out


def fillet_inside(shape, r, up=(0, 0, 1), keep=lambda c: True, eps=0.05):
    """Fillet every inside (concave) corner selected by keep(edge centre), radius r, except corners that
    touch a face pointing down in the print orientation (`up` is the print +z in this part's frame):
    a fillet there would add an overhang. Returns (shape, number of edges filleted)."""
    from build123d import Vector
    up = Vector(*up)
    solid = max(shape.solids(), key=lambda x: x.volume) if hasattr(shape, "solids") else shape   # not [0]: a stray
    # fragment from a boolean can come first (Nov 22: the platter came out as a 95 mm3 socket core 3 runs in 4)
    picked = []
    for e, fs in _edge_faces(solid):
        if len(fs) != 2:
            continue
        p = e.position_at(0.5)
        if not keep(p):
            continue
        n1, n2 = fs[0].normal_at(p), fs[1].normal_at(p)
        if abs(n1.dot(n2)) > 0.98:                         # tangent (smooth) edge
            continue
        if min(n1.dot(up), n2.dot(up)) < -0.2:              # touches a ceiling in print orientation
            continue
        d = (n1 - n2).normalized() * eps
        if solid.is_inside(p + d) and solid.is_inside(p - d):
            picked.append(e)
    if not picked:
        return solid, 0
    try:
        out = solid.fillet(r, picked)
    except Exception:                                       # fall back one edge at a time, skip failures
        out, n = solid, 0
        for e in picked:
            try:
                out = out.fillet(r, [min(out.edges(), key=lambda x: (x.position_at(0.5) - e.position_at(0.5)).length)])
                n += 1
            except Exception:
                pass
        return out, n
    return out, len(picked)


FILLETS = {}               # part -> edges filleted, filled as the parts are built (for the report)


def platter():
    """Assembly orientation: z = 0 is the face that sits on the horn."""
    p = Pos(0, 0, HUB_H - PLATE_T) * Cylinder(PLAT_R, PLATE_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    p += Cylinder(HUB_R, HUB_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    rib_z = HUB_H - PLATE_T - RIB_H
    for k in range(8):
        a = 22.5 + 45 * k
        rib = Pos((HUB_R + PLAT_R - 3) / 2, 0, rib_z) * Box(PLAT_R - 3 - HUB_R + 1, 2.4, RIB_H,
                                                            align=(Align.CENTER, Align.CENTER, Align.MIN))
        p += Rot(Z=a) * rib
    rim = Pos(0, 0, rib_z) * Cylinder(PLAT_R, RIB_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    rim -= Pos(0, 0, rib_z) * Cylinder(PLAT_R - 2.5, RIB_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    p += rim
    # horn: centring recess, centre-screw pocket, four radial slots with counterbores from the top
    p -= Pos(0, 0, -1) * Cylinder(HORN_OD / 2 + 0.5, 1.8, align=(Align.CENTER, Align.CENTER, Align.MIN))   # cutters overshoot
    p -= Pos(0, 0, -1) * Cylinder(4.5, 5.0, align=(Align.CENTER, Align.CENTER, Align.MIN))                 # the faces they open
    for k in range(4):
        p -= Rot(Z=45 + 90 * k) * slot(5.0, 9.0, HORN_SCREW, HUB_H)
        p -= Rot(Z=45 + 90 * k) * slot(5.0, 9.0, CB_D, HUB_H - SEAT_Z, z0=SEAT_Z)
    # spindle socket: straight through into the centre-screw pocket. Printed top face down, a blind socket's floor
    # was a 5 mm bridge that sagged and stopped the spindle seating (Nov 22); through, the hole opens into the
    # wider pocket and has no ceiling. 5.4 mm for the spindle's 5.0 pin: small holes print 0.2-0.4 mm under.
    # The cutter overshoots both faces: flush with the top and the underside, OCC returned the socket's core instead
    # of the platter (Nov 22).
    p -= Pos(0, 0, -1) * Cylinder(SOCKET_D / 2, HUB_H + 2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    p -= Pos(0, 0, HUB_H - 0.7) * Cone(SOCKET_D / 2, SOCKET_D / 2 + 1.0, 1.0, align=(Align.CENTER, Align.CENTER, Align.MIN))  # chamfer, 45 deg, overshooting the top
    p = max(p.solids(), key=lambda x: x.volume)    # the coaxial socket cuts leave a 95 mm3 fragment in the hole
    # inside corners at the hub and the rim: hub and rim into the plate, rib ends into the hub and rim
    # (printed top face down). The rib roots along the plate stay sharp: where a root fillet meets the hub
    # and rim fillets the mesher leaves cracks, and the ribs only stiffen the plate; the torque goes
    # through the hub.
    ring = lambda c: abs(math.hypot(c.X, c.Y) - HUB_R) < 0.3 or abs(math.hypot(c.X, c.Y) - (PLAT_R - 2.5)) < 0.3
    p, n = fillet_inside(p, 1.5, up=(0, 0, -1), keep=ring)
    FILLETS["platter"] = n
    sol = p.solids()
    assert len(sol) == 1 and sol[0].volume > 80000, ("platter boolean failed", [round(x.volume) for x in sol])
    return p


def spindle(d=SPINDLE_D, marks=0):
    """Record spindle in assembly orientation (platter top at z = 0), which is also how it prints: key on the bed,
    no overhangs. d: diameter above the platter; marks: dimples on the top face that tell the fit set apart."""
    h = HUB_H - KEY_Z0
    MIN = (Align.CENTER, Align.CENTER, Align.MIN)
    key = Cylinder(SOCKET_D / 2 - KEY_CLR, h, align=MIN)
    for k in range(4):
        key += Rot(Z=45 + 90 * k) * slot(5.0, 9.0, CB_D - 2 * KEY_CLR, h)
    s = Pos(0, 0, -h) * key
    s += Cylinder(d / 2, SPINDLE_UP, align=MIN)                                            # above the platter
    s += Pos(0, 0, SPINDLE_UP) * Cone(d / 2, d / 2 - 0.8, 0.8, align=MIN)                  # lead-in chamfer
    # no fillet at the column's root: it would stand above the platter top, where the record's hole sits
    for k in range(marks):
        a = math.radians(360 * k / marks)
        rr = 1.4 if marks > 1 else 0.0
        s -= Pos(rr * math.cos(a), rr * math.sin(a), SPINDLE_UP + 0.8 - 0.4) * Cylinder(0.45, 1.0, align=MIN)
    solids = s.solids()
    assert len(solids) == 1, ("spindle came out in pieces", len(solids))
    return solids[0]


# ---- base -------------------------------------------------------------------------------
# Servo underside and height stack from the SO-ARM100 STEP (vendor/STS3215_03a.step, Nov 21), servo frame
# with the horn axis at x = 0 and the cover plate (what the servo stands on) at z = 0:
#   cover plate |y| < 9.9 from 5.3 mm behind the port end to the horn end; the rim outside it is 1.8-3.3 mm higher
#   bearing boss under the horn axis, r 3.0, 0.6 mm below the cover
#   two 3-pin sockets facing down, pins at x -14.9..-13.1, |y| < 8.6, tips 1.7 mm below the cover
#   case top 34.7 above the cover (the datasheet's 35), horn top 36.4, centre screw head 37.9
SEAT = 14.5               # cover plate height: the pin tips are 12.8 mm above the desk, >= 12.5 for the plugs (measured)
FLOOR = SEAT              # name kept for the pocket and the stand-in
COVER_HW, COVER_X0 = 9.9, -(SERVO_L - SHAFT_FROM_END) + 5.3
BOSS_R, BOSS_DOWN = 3.0, 0.6
PIN_X, PIN_HW, PIN_DOWN = (-14.9, -13.1), 8.6, 1.7
COVER_TO_HORN = 36.4
Z_HORN = SEAT + COVER_TO_HORN
PLUG_X = (-21.0, -7.0)    # floor opening under the sockets, desk up: the plugs and their wires hang here
PLUG_HW = 11.5
CHANNEL_Z = SEAT - 5.0    # cable channel from the opening out through the port end, under a bridge that holds the cover
POCKET_H = 22.0
TOWER_OD, TOWER_BORE = 24.0, 10.7      # bore: the 10.0 sleeve slides in
# Pivot pin: a whole 1.1 mm T-pin, no cutting (Nov 21). It stands point-up with its T bar in a slot at the
# bottom of the tower bore, on printed shims, so the arm's weight goes straight down the pin into the base. The
# peg is a sleeve that slides down over the pin into the bore and sits on the T bar, holding the pin upright.
# Shims under the T bar raise the pin, the sleeve and the arm together, so the 3 mm gap under the hub stays.
PIN_LEN, PIN_WIRE = 44.3, 1.2     # point to the back of the T (measured), wire diameter 1.1-1.2 (measured, Nov 22): the larger drawn
T_BAR, T_THICK = 14.4, 2.2        # T head: bar length, depth along the pin (measured, Nov 21)
SHIM_L, SHIM_W = 14.8, 2.2    # as first printed (Nov 22); the slot is drawn around them
SLOT_L, SLOT_W = T_BAR + 1.4, 2.9 # slot through the tower floor region, along x, open to the top for the T bar (and the 14.8 x 2.2 shims)
SHIM_NOM, SHIMS = 3.0, (1.0, 2.0, 4.0)   # nominal stack 1 + 2 mm: the arm can go 3 mm lower or 4 mm higher
NEEDLE_OUT = 13.0          # pin tip above the sleeve top: the sleeve must stay below the hub (10 mm under the tip)
PEG_D = 10.0
PEG_L = PIN_LEN - T_THICK - NEEDLE_OUT
NEEDLE_HOLE = 2.1          # sleeve bore for the 1.1-1.2 mm pin, prints upright; 1.4 and 1.6 both printed too tight (Nov 22): holes this small shrink up to ~0.5 mm
OUT_X, OUT_RAIL_Y = -84.0, 40.0   # outrigger cross bar outer edge (x) and rail centres (+-y)
# Shell interface (for a bolt-on case; not used by the spine build): six M3 heat-set insert bosses on the ends of the three cross bars,
# tops on one datum plane, for a future deck/skirt/collar/electronics shell that bolts on over this frame.
SHELL_BOSS_OD, SHELL_BOSS_UP = 8.0, 2.0          # boss diameter, height above the 6 mm strips (datum z = 8)
SHELL_INSERT_D, SHELL_INSERT_DEPTH = 4.0, 6.0    # hole for M3 x 4 mm heat-set inserts (2 mm floor left)
FOOT_H = 6.0              # beam, feet and anchor strip: one height, so the tower root is one plane (clean fillet)


def base():
    x_far = SHAFT_FROM_END + FIT                       # pocket edge toward the tonearm
    x_near = -(SERVO_L - SHAFT_FROM_END) - FIT
    blk = Pos((x_near + x_far) / 2, 0, 0) * Box(x_far - x_near + 8, SERVO_W + 2 * FIT + 8, FLOOR + POCKET_H,
                                                  align=(Align.CENTER, Align.CENTER, Align.MIN))
    pocket = Pos((x_near + x_far) / 2, 0, FLOOR) * Box(x_far - x_near, SERVO_W + 2 * FIT, POCKET_H,
                                                       align=(Align.CENTER, Align.CENTER, Align.MIN))
    blk -= pocket
    # The two sockets are on the servo's underside, toward the end away from the horn, plugs pointing down.
    # The floor carries the cover plate in two places: under the horn half (with a recess for the bearing
    # boss) and on a bridge under the port end. Between them an opening from the desk up takes the plugs,
    # and a channel under the bridge takes the cable out through the open port end (cut below, after the
    # feet). The horn-end wall stays (the shaft position is set from it); side walls and zip ties hold it.
    blk -= Pos(x_near - 2.5, 0, 0) * Box(5, SERVO_W + 2 * FIT, FLOOR + POCKET_H + 1, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for y in (-1, 1):                                  # zip-tie slots across the pocket walls
        blk -= Pos(-10, y * (SERVO_W / 2 + FIT + 2), FLOOR + 14) * Box(4, 6, 2.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b = blk
    beam_x0 = x_far + 4
    b += Pos((beam_x0 + ARM_D) / 2, 0, 0) * Box(ARM_D - beam_x0, 20, FOOT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # feet so it does not tip: a cross bar under the servo block and one under the tower, both 140 mm wide
    b += Pos(x_near + 6, 0, 0) * Box(12, 140, FOOT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b += Pos(ARM_D, 0, 0) * Box(14, 140, FOOT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # outrigger under the far side of the platter: pressing the rim there (putting a record on) would
    # otherwise tip the base over the servo-end foot at about 350 g. Two rails, centre left clear for the
    # servo plugs and cable, and a cross bar under the rim.
    for y in (-OUT_RAIL_Y, OUT_RAIL_Y):
        b += Pos((OUT_X + x_near + 12) / 2, y, 0) * Box(x_near + 12 - OUT_X, 10, FOOT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b += Pos(OUT_X + 5, 0, 0) * Box(10, 2 * OUT_RAIL_Y + 10, FOOT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # shell mounting bosses (added before the cuts so the plug opening cannot clip them; none sit near it)
    for (bx, by) in shell_points():
        b += Pos(bx, by, 0) * Cylinder(SHELL_BOSS_OD / 2, FOOT_H + SHELL_BOSS_UP, align=(Align.CENTER, Align.CENTER, Align.MIN))
        b -= Pos(bx, by, FOOT_H + SHELL_BOSS_UP - SHELL_INSERT_DEPTH) * Cylinder(SHELL_INSERT_D / 2, SHELL_INSERT_DEPTH,
                                                                                align=(Align.CENTER, Align.CENTER, Align.MIN))
    # electronics tray (servo board pad + UNO on standoffs) and cable loops; USB/power at the tower end
    (tx0, ty0), (tx1, ty1), _ = ELECTRONICS_BAY
    b += Pos((tx0 + tx1) / 2, (ty0 + ty1) / 2, 0) * Box(tx1 - tx0, ty1 - ty0, TRAY_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    sd, sh, sp = UNO_STANDOFF
    for hx, hy in uno_holes_xy():
        b += Pos(hx, hy, TRAY_T) * Cylinder(sd / 2, sh, align=(Align.CENTER, Align.CENTER, Align.MIN))
        b -= Pos(hx, hy, TRAY_T + sh - 6.0) * Cylinder(sp / 2, 6.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for lx, ly, lz, ax_ in cable_loops():
        b += Pos(lx, ly, lz) * cable_loop(ax_)
    # plug opening (desk to above the floor) and cable channel (desk to CHANNEL_Z, out through the open end);
    # they also cut the centre of the servo foot bar
    b -= Pos(sum(PLUG_X) / 2, 0, 0) * Box(PLUG_X[1] - PLUG_X[0], 2 * PLUG_HW, SEAT + 1,
                                          align=(Align.CENTER, Align.CENTER, Align.MIN))
    b -= Pos((x_near - 6 + PLUG_X[0] + 1) / 2, 0, 0) * Box(PLUG_X[0] + 1 - (x_near - 6), 2 * PLUG_HW, CHANNEL_Z,
                                                           align=(Align.CENTER, Align.CENTER, Align.MIN))
    b -= Pos(0, 0, SEAT - BOSS_DOWN - 0.6) * Cylinder(BOSS_R + 0.6, BOSS_DOWN + 1.6,
                                                     align=(Align.CENTER, Align.CENTER, Align.MIN))   # boss recess
    # pivot tower
    b += Pos(ARM_D, 0, 0) * Cylinder(TOWER_OD / 2, TOWER_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b -= Pos(ARM_D, 0, FLOOR_Z) * Cylinder(TOWER_BORE / 2, TOWER_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b -= Pos(ARM_D, 0, FLOOR_Z) * Box(SLOT_L, SLOT_W, TOWER_H, align=(Align.CENTER, Align.CENTER, Align.MIN))   # T bar slot
    # wire anchor: a fin straight below the arm's drop hole at mid sweep, tie hole near the top
    ax, ay = anchor_xy()
    phi = math.degrees(math.atan2(ay, ax - ARM_D))
    fin = Box(3, 10, ANCHOR_Z + 4, align=(Align.CENTER, Align.CENTER, Align.MIN))
    fin -= Pos(0, 0, ANCHOR_Z) * Rot(Y=90) * Cylinder(1.5, 4)
    reach = math.hypot(ax - ARM_D, ay)
    b += Pos(ARM_D, 0, 0) * Rot(Z=phi) * Pos(reach / 2 + 2, 0, 0) * Box(reach + 4, 10, FOOT_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    b += Pos(ax, ay, 0) * Rot(Z=phi) * fin
    # arm rest: a fork under the beam REST_D from the pivot with the stylus parked at REST_R, off the platter.
    # The stylus-heavy arm tips down until the beam sits on the fork floor, REST_DROP under its playing
    # height, so the stylus hangs over the desk instead of falling onto the platter; the walls stop it
    # swinging back. Lift it over a wall to play.
    rest = rest_frame()
    post = Box(REST_LEN, REST_W, REST_TOP, align=(Align.CENTER, Align.CENTER, Align.MIN))
    post -= Pos(0, 0, REST_TOP - REST_WALL) * Box(REST_LEN + 1, BEAM + 2 * REST_GAP, REST_WALL + 1,
                                                  align=(Align.CENTER, Align.CENTER, Align.MIN))
    s0 = TOWER_BORE / 2 + 1.0                          # strip from the tower (outside its bore) to the post, fused to the post first
    post += Pos((s0 - REST_D + REST_LEN / 2) / 2, 0, 0) * Box(REST_D - s0 + REST_LEN / 2, 10, FOOT_H,
                                                 align=(Align.CENTER, Align.CENTER, Align.MIN))
    b += rest * post
    assert len(b.solids()) == 1, "arm rest did not fuse to the base"
    # inside corners: tower root, fin root, beam and feet into the servo block. Not the pocket, its cable
    # exits and zip-tie slots (fits), not the tower bore and pin hole, not the rest fork (the beam sits in it).
    bx0, bx1 = x_near - 4 + 0.3, x_far + 4 - 0.3
    by = (SERVO_W + 2 * FIT + 8) / 2 - 0.3
    in_block = lambda c: bx0 < c.X < bx1 and abs(c.Y) < by
    in_tower = lambda c: math.hypot(c.X - ARM_D, c.Y) < TOWER_OD / 2 - 0.3 and c.Z > 5.5
    near_fin = lambda c: math.hypot(c.X - ax, c.Y - ay) < 9
    cable_exit = lambda c: abs(c.X - (x_far + 4)) < 1.5 and abs(c.Y) < 7.5
    b, n1 = fillet_inside(b, 1.5, keep=lambda c: near_fin(c))
    in_fork = lambda c: c.Z > REST_TOP - REST_WALL - 0.5
    near_boss = lambda c: min(math.hypot(c.X - px, c.Y - py) for px, py in shell_points()) < SHELL_BOSS_OD / 2 + 1
    in_tray = lambda c: tx0 - 0.5 < c.X < tx1 + 0.5 and ty0 - 0.5 < c.Y < ty1 + 0.5 and c.Z < TRAY_T + UNO_STANDOFF[1] + 0.5
    near_loop = lambda c: min(math.hypot(c.X - lx, c.Y - ly) for lx, ly, _, _ in cable_loops()) < 8
    b, n2 = fillet_inside(b, 3.0, keep=lambda c: not (in_block(c) or in_tower(c) or near_fin(c) or cable_exit(c)
                                                       or in_fork(c) or near_boss(c) or in_tray(c) or near_loop(c)))
    FILLETS["base"] = n1 + n2
    return b


def peg():
    """Sleeve over the pivot pin, assembly orientation, z up (prints upright: round bore, no overhang)."""
    p = Cylinder(PEG_D / 2, PEG_L, align=(Align.CENTER, Align.CENTER, Align.MIN))
    p -= Cylinder(NEEDLE_HOLE / 2, PEG_L, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return p


def tpin():
    """T-pin stand-in: back of the T at z = 0, point at PIN_LEN, T bar along x."""
    t = Box(T_BAR, PIN_WIRE, T_THICK, align=(Align.CENTER, Align.CENTER, Align.MIN))
    t += Cylinder(PIN_WIRE / 2, PIN_LEN - 1.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    t += Pos(0, 0, PIN_LEN - 1.5) * Cone(PIN_WIRE / 2, 0.05, 1.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return t


def shim(t):
    """Bar that drops into the tower slot under the T bar."""
    return Box(SHIM_L, SHIM_W, t, align=(Align.CENTER, Align.CENTER, Align.MIN))


# ---- tonearm (arm frame: pivot at origin, stylus at (ARM_L, 0), bed at z = 0) ----------
BEAM = 8.0
HUB = 20.0
HUB_T = 16.0
CONE_H = 10.0             # pivot point height above the arm's underside (the needle tip; name kept from the printed cone)
# Pivot cup: an M3x6 socket head cap screw (ISO 4762, from the SO-101 kit) fixed head-down in the hub. The needle
# point sits in the drill-point cone at the bottom of its 2.5 mm hex socket: steel on steel, so the point cannot
# dig a pit the way it would in PLA (0.45 N on a 0.05 mm point is about 60 MPa, PLA's yield stress).
CUP_HEAD_D, CUP_HEAD_H, CUP_LEN = 5.5, 3.0, 6.0
CUP_SOCKET_T = 1.6        # socket depth to the cone bottom, about the ISO 4762 minimum key engagement (1.3) plus the point
CUP_FACE = CONE_H - CUP_SOCKET_T                   # head face (socket opening) height, facing down
CUP_BORE_D = 6.2          # pivot counterbore: the 5.7 button head fits (5.8 printed too tight, Nov 22)
# What is actually fitted (Nov 22): the kit on hand is BESITU ISO 7380 button heads, no ISO 4762 cap screws. An M3x8
# button head (head 5.7 x 1.65, 2.0 mm hex socket, ~1.2 to the cone bottom) fits the 5.8 counterbore and pulls up to
# its top, so its face sits 1.35 above CUP_FACE and the point's seat is 1.0 above CONE_H. The printed arm and tower
# are unchanged: the pin goes 1.0 higher on the single 4 mm shim instead of 1 + 2, and the stylus height is as designed.
PIVOT_HEAD_D, PIVOT_HEAD_K, PIVOT_SOCKET_T, PIVOT_LEN = 5.7, 1.65, 1.2, 8.0
PIVOT_FACE = CUP_FACE + CUP_HEAD_H - PIVOT_HEAD_K                        # 1.35 above CUP_FACE
SHIM_USE = 4.0                                                           # the 4 mm shim alone (SHIM_NOM + 1)
PIVOT_RISE = SHIM_USE - SHIM_NOM          # seat computed 0.95 above CONE_H; the 4 mm shim lifts the pin 1.0 (0.05 high)
STUB_L = 80.0
SHELL_T = 3.0
MOUNT = 12.7              # 1/2 in standard cartridge mount
MOUNT_SLOT = (5.0, 15.0)  # slot centre range behind the stylus, along the cartridge axis
SCREW_SLOT_W = 3.2        # passes M2, M2.5 or M3 (the SO-101 kit has M2x6 and M3x6)
TIE_X = (118.0,)          # strain-relief tab behind the headshell; the leads run inside the beam (WIRE_TUNNEL)
WIRE_TUNNEL_D, WIRE_EXIT_X = 2.6, 14.0   # tunnel along the beam axis, side exit on the -y face just past the hub
DROP = (-18.0, -16.0)         # wire drop hole beside the stub, 24.1 mm from the pivot axis: a 45-50 mm drop span can
                              # bow any way and still clear the tower by 3 mm (was 15.3 mm and touched)
# Cartridge: CKANDAY "universal" = a Chuo Denshi CZ-800 clone, snap-in, NO 1/2 in screw holes (Nov 21).
# Measured with calipers (Nov 22): body 9.5 wide, 9.4 tall at its largest; 30.8 from the front of the red
# stylus to the pin tips; the black housing 28.8 from its front to the pin tips (so the red stylus stands 2.0 ahead
# of it); pins 3.8 past the rear face; stylus tip 2.0 behind the red front and 9.9 below the top face (0.5 below
# the body's bottom). From his photos (scaled by the 30.8 reading, +-1 mm): the red stylus covers the first 14.4 mm
# of the underside and pulls off forward. 2.1 g is the published CZ-800 mass (not weighed).
# FreeMason's CZ-800 clip (vendor/cz800_clip_freemason) does not fit it on this arm (Nov 22): it holds the body by
# snap features between a rear wall and a front bar 27 mm apart, its 1/2 in notches graze the body, and nothing
# can sit beside the body to hold a screw. It is replaced by cart_mount() below.
CART_W, CART_T = 9.5, 9.4                # body width and height
CART_L = 27.0                            # red front to the rear face
CART_FRONT = 2.0                         # red front ahead of the stylus tip
CART_TIP_DROP = 9.9 - CART_T             # tip below the body's bottom: 0.5
CART_RED = 14.4                          # red stylus length on the underside, from its front (photo, +-1)
CART_PIN_L = 3.8
MOUNT_T = 3.5                            # mount plate: 3.5 + 9.9 = 13.4, the stylus drop the arm was designed for
MOUNT_SCREW_X = 10.0                     # mount screw line behind the stylus: the middle of the headshell slots (5-15)
MOUNT_PILOT = 1.7                        # M2 self-taps into PLA from a 1.7 mm hole
CHEEK_X = (15.5, 24.0)                   # side cheeks on the black part only, behind the red stylus
CHEEK_T, CHEEK_H = 1.6, 6.0
CLAMP_GAP = 0.6                          # body to the clamp-side cheek; the clamp screw takes it up
CART_H = MOUNT_T + CART_T + CART_TIP_DROP    # stylus tip to headshell underside: 13.4


def headshell_sign():
    """Which way to turn the headshell so the cartridge lies along the groove, in the arm's design frame (stylus on
    +y; arm_side() mirrors the finished parts when ARM_SIDE is -1, which flips the turn with them)."""
    r = 50.19
    xs = (ARM_D ** 2 + r ** 2 - ARM_L ** 2) / (2 * ARM_D)
    S = np.array([xs, math.sqrt(r * r - xs * xs)])
    P = np.array([ARM_D, 0.0])
    rear = (P - S) / np.linalg.norm(P - S)                   # stylus -> pivot
    t = np.array([-S[1], S[0]]) / r                          # groove tangent
    best = None
    for sgn in (1, -1):
        a = math.radians(sgn * ARM_BETA)
        c = np.array([rear[0] * math.cos(a) - rear[1] * math.sin(a), rear[0] * math.sin(a) + rear[1] * math.cos(a)])
        e = abs(abs(float(np.dot(c, t))) - 1.0)
        if best is None or e < best[0]:
            best = (e, sgn)
    # in the arm frame the stylus->pivot direction is -x; the frames differ by a rotation,
    # which preserves the sense of the turn
    return best[1]


def teardrop_x(r, length):
    """Horizontal hole along x, centred on the origin: circle r with a 45 deg peak on top (printable roof)."""
    t = Rot(Y=90) * Cylinder(r, length)
    t += Pos(0, 0, r / math.sqrt(2)) * Rot(X=45) * Box(length, r, r)   # corners on the circle, peak at r*sqrt(2)
    return t


HEADSHELL_PAD = None      # set by headshell.install(): the arm then ends in a pad for the separate headshell


def arm():
    sgn = headshell_sign()
    a = Box(HUB, HUB, HUB_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # pivot cup: funnel from the underside, head counterbore, 2.5 mm pilot for the M3 thread (self-taps in PLA)
    a -= Cone(5.0, CUP_BORE_D / 2, CUP_FACE, align=(Align.CENTER, Align.CENTER, Align.MIN))
    a -= Pos(0, 0, CUP_FACE) * Cylinder(CUP_BORE_D / 2, CUP_HEAD_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    a -= Pos(0, 0, CUP_FACE + CUP_HEAD_H) * Cylinder(1.25, HUB_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    beam_end = ARM_L - 24.0       # stop behind the mount slots; the headshell carries the rest
    a += Pos((HUB / 2 + beam_end) / 2, 0, 0) * Box(beam_end - HUB / 2 + 1, BEAM, BEAM,
                                                    align=(Align.CENTER, Align.CENTER, Align.MIN))
    a += Pos(-(HUB / 2 + STUB_L) / 2, 0, 0) * Box(STUB_L - HUB / 2 + 1, BEAM, BEAM,
                                                  align=(Align.CENTER, Align.CENTER, Align.MIN))
    # wire management: tie tabs on the -y side of the beam, the last one is the strain relief behind
    # the headshell; a drop tab beside the stub lets the leads fall clear of the peg (DROP from the axis)
    for x in TIE_X:
        a += Pos(x, -(BEAM / 2 + 3), 0) * Box(6, 6, 3, align=(Align.CENTER, Align.CENTER, Align.MIN))
        a -= Pos(x, -(BEAM / 2 + 3.5), 0) * Cylinder(1.0, 3, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # wire tunnel: in at the beam's end behind the cartridge, out the -y side next to the hub, then to the drop
    # hole. A 2.6 mm horizontal hole in the 8 mm beam prints fine; feed the 30 AWG pair with a thread.
    # Teardrop section (round below, 45 deg roof above) so the holes print with no overhang.
    t_len = beam_end - WIRE_EXIT_X + 1
    a -= Pos(WIRE_EXIT_X + t_len / 2, 0, BEAM / 2) * teardrop_x(WIRE_TUNNEL_D / 2, t_len)
    # the side exit goes right through (Nov 22): a lead pushed along the tunnel jams at the 90 deg corner, and the
    # opposite hole lets a pin or stiff wire in from the +y side to guide or hook it out of the -y exit
    a -= Pos(WIRE_EXIT_X, 0, BEAM / 2) * Rot(Z=90) * teardrop_x(WIRE_TUNNEL_D / 2, BEAM + 1.0)
    tab_y0, tab_y1 = -BEAM / 2 + 1, DROP[1] - 4                 # from inside the stub out past the hole
    a += Pos(DROP[0], (tab_y0 + tab_y1) / 2, 0) * Box(8, tab_y0 - tab_y1, 3, align=(Align.CENTER, Align.CENTER, Align.MIN))
    a -= Pos(*DROP, 0) * Cylinder(1.5, 3, align=(Align.CENTER, Align.CENTER, Align.MIN))
    # headshell in the cartridge frame: origin at the stylus, +x' rearward along the cartridge
    if HEADSHELL_PAD is not None:                    # headshell.py: a pad the separate headshell bolts to
        shell = HEADSHELL_PAD()
    else:                                            # the 1/2 in slotted plate
        shell = Pos(12.0, 0, 0) * Box(36, 26, SHELL_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
        shell += Pos(2.0, 13 + 7, 0) * Box(10, 14, SHELL_T, align=(Align.CENTER, Align.CENTER, Align.MIN))   # finger lift
        for y in (-MOUNT / 2, MOUNT / 2):
            shell -= Pos(0, y, 0) * slot(*MOUNT_SLOT, SCREW_SLOT_W, SHELL_T)
    # +x' points from the stylus back toward the pivot (arm -x), turned by the offset angle
    shell = Pos(ARM_L, 0, 0) * Rot(Z=180 + sgn * ARM_BETA) * shell
    a += shell
    # inside corners (printed flat, so every one of these faces up or sideways): 1 mm at the small tabs,
    # 2 mm where the beam and stub meet the hub and where the beam meets the headshell. Not the pivot cone.
    tab = lambda c: c.X < 110 and c.Y < -BEAM / 2 + 0.05 and c.Z < 3.1
    a, n1 = fillet_inside(a, 1.0, keep=lambda c: tab(c) and math.hypot(c.X, c.Y) > 7)
    a, n2 = fillet_inside(a, 2.0, keep=lambda c: not tab(c) and math.hypot(c.X, c.Y) > 7)
    FILLETS["arm"] = n1 + n2
    return a


WT_TONGUE_W, WT_TONGUE_L, WT_TONGUE_T, WT_BUMP = 8.0, 7.5, 1.0, 0.45   # counterweight friction tongue
WEIGHT_Y = 3.2            # cup centre beside the stub, +y: balances the headshell offset and the leads (balance.py)


def weight():
    """Carriage: channel for the 8 mm stub on top, open cup for coins/nuts below, offset WEIGHT_Y to +y
    so the arm hangs level in roll (the cartridge and the leads sit on the -y side)."""
    L = 22.0
    W = 2 * (WEIGHT_Y + BEAM / 2 + 0.3 + 2.0)       # keeps a 2 mm wall beside the channel on the -y side
    w = Pos(0, WEIGHT_Y, 0) * Box(L, W, 30, align=(Align.CENTER, Align.CENTER, Align.MIN))
    w -= Pos(0, 0, 30 - 2 - BEAM - 0.5) * Box(L, BEAM + 0.6, BEAM + 0.6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    w -= Pos(0, WEIGHT_Y, 2) * Box(L - 4, W - 4, 16, align=(Align.CENTER, Align.CENTER, Align.MIN))
    w -= Pos(0, WEIGHT_Y + W / 2, 2 + 4) * Box(L - 8, 6, 12, align=(Align.CENTER, Align.CENTER, Align.MIN))   # loading window
    # friction tongue (Nov 22: the carriage printed loose on the stub). The channel's -y wall gets a tongue,
    # attached at the top and free at the bottom, whose bump presses the stub's side: it holds the carriage where
    # it is set and still slides by hand. Vertical slots and a short gap only, so it prints roof-down unsupported.
    z0 = 30 - 2 - BEAM - 0.5                                         # channel floor
    yw = -(BEAM + 0.6) / 2                                           # the wall's inner face
    tx, t_top = WT_TONGUE_W / 2, z0 + WT_TONGUE_L
    box = lambda x0, x1, y0, y1, za, zb: Pos((x0 + x1) / 2, (y0 + y1) / 2, za) * Box(x1 - x0, y1 - y0, zb - za, align=(Align.CENTER, Align.CENTER, Align.MIN))
    yo = WEIGHT_Y - W / 2                                            # the wall's outer face
    for xa in (-tx - 0.6, tx):
        w -= box(xa, xa + 0.6, yo - 1, yw + 0.01, z0 - 0.5, t_top)  # slits either side
    w -= box(-tx, tx, yo - 1, yw + 0.01, z0 - 0.5, z0 + 0.1)          # gap under its free end
    w -= box(-tx, tx, yo - 1, yw - WT_TONGUE_T, z0 - 0.5, t_top)     # thinned to WT_TONGUE_T
    from build123d import Plane, Polygon, extrude
    b = WT_BUMP
    prof = [(tx - 0.5, yw), (tx - 0.5 - b, yw + b), (-tx + 0.5 + b, yw + b), (-tx + 0.5, yw)]   # ramped ends; counter-clockwise, so it extrudes up
    w += Pos(0, 0, z0 + 0.1) * extrude(Polygon(*prof, align=None), 3.0)
    return w


# ---- assembly pose ------------------------------------------------------------------------
ANCHOR_Z = Z_HORN - 12.6   # tie hole height on the anchor fin
Z_REC = Z_HORN + HUB_H + 1.2                     # record top
Z_ARM = Z_REC + CART_H                           # arm frame z = 0 (headshell underside)
TIP_Z = Z_ARM + CONE_H                           # pivot pin point
FLOOR_Z = TIP_Z - PIN_LEN - SHIM_NOM             # tower floor under the shims
TOWER_H = TIP_Z - NEEDLE_OUT - 8.0               # tower top: 8 mm of sleeve shows above it


def stylus_xy(r):
    xs = (ARM_D ** 2 + r * r - ARM_L ** 2) / (2 * ARM_D)
    return np.array([xs, ARM_SIDE * math.sqrt(max(r * r - xs * xs, 0.0))])


def arm_angle(r):
    """Arm rotation (deg, about the pivot) that puts the stylus at radius r."""
    S = stylus_xy(r)
    return math.degrees(math.atan2(S[1], S[0] - ARM_D))


def anchor_xy():
    a = math.radians(arm_angle(sum(GROOVE_R) / 2))
    dx, dy = DROP[0], ARM_SIDE * DROP[1]                # DROP is in the design frame
    return np.array([ARM_D + dx * math.cos(a) - dy * math.sin(a), dx * math.sin(a) + dy * math.cos(a)])


# arm rest (built into the base)
REST_R = 130.0            # stylus radius when parked: the fork stays 4 mm clear of the beam at r 90 (record edge)
REST_D = 55.0             # fork centre from the pivot, between the tie tabs at 40 and 80
REST_DROP = 1.0           # fork floor under the beam's playing height: the arm tips about 1 deg onto it
REST_WALL, REST_GAP, REST_LEN = 6.0, 1.5, 10.0
REST_W = BEAM + 2 * REST_GAP + 2 * 3.0
REST_TOP = Z_ARM - REST_DROP + REST_WALL
REST_ANGLE = arm_angle(REST_R)


def rest_frame():
    return Pos(ARM_D, 0, 0) * Rot(Z=REST_ANGLE) * Pos(REST_D, 0, 0)


# ---- shell interface (bolt-on case) ------------------------------------------------------------------
def shell_points():
    """(x, y) of the six M3 insert bosses, turntable frame (spindle at the origin, desk z = 0); tops at
    SHELL_DATUM_Z. Ends of the servo-end and tower cross bars (y +-64) and of the outrigger cross bar (y +-40)."""
    x_near = -(SERVO_L - SHAFT_FROM_END) - FIT
    return [(x_near + 6, -64.0), (x_near + 6, 64.0), (ARM_D, -64.0), (ARM_D, 64.0),
            (OUT_X + 5, -OUT_RAIL_Y), (OUT_X + 5, OUT_RAIL_Y)]


SHELL_DATUM_Z = FOOT_H + SHELL_BOSS_UP
PLATTER_RIM_UNDERSIDE = Z_HORN + HUB_H - PLATE_T          # plate underside at the rim (ribs stop 3 mm inside it)
PLATTER_RIB_UNDERSIDE = Z_HORN + HUB_H - PLATE_T - RIB_H  # lowest spinning point, inside r PLAT_R - 3
DECK_TOP_MAX_OUTSIDE = PLATTER_RIM_UNDERSIDE - 3.0        # deck top outside r PLAT_R + 3
DECK_TOP_MAX_UNDER = PLATTER_RIB_UNDERSIDE - 3.0          # anything reaching under the platter
ELECTRONICS_BAY = ((20.0, -72.0), (125.0, -10.0), 30.0)   # tray (x0, y0), (x1, y1) and clear height above the desk
TRAY_T = 2.0                                              # tray plate, fused to the beam's -y side
# Arduino UNO R3: 68.58 x 53.34 board, M3 holes at (13.97, 2.54), (15.24, 50.8), (66.04, 7.62), (66.04, 35.56) from
# the corner at the USB/power edge (Arduino reference drawing). Turned 180 deg so USB and power face the tower end.
UNO_W, UNO_H = 68.58, 53.34
UNO_HOLES = [(13.97, 2.54), (15.24, 50.8), (66.04, 7.62), (66.04, 35.56)]
UNO_ORIGIN = (125.0 - 2.0, -72.0 + 1.0)                   # board corner at the USB edge (x max), lower edge (y min)
UNO_STANDOFF = (6.0, 5.0, 2.5)                            # diameter, height above the tray, M3 self-tap pilot
SERVO_BOARD_PAD = ((22.0, -70.0), (50.0, -14.0))          # flat area for the bus servo board (tape it; size unknown)


def uno_holes_xy():
    ox, oy = UNO_ORIGIN
    return [(ox - hx, oy + hy) for hx, hy in UNO_HOLES]  # 180 deg about z: x runs back from the USB edge


def cable_loops():
    """(x, y, z_base, channel axis 'x'|'y') of the printed cable loops (5 x 6 mm channels)."""
    x_near = -(SERVO_L - SHAFT_FROM_END) - FIT
    return [(x_near + 6, -30.0, FOOT_H, "y"),             # servo cable along the servo-end foot bar
            (24.0, -40.0, TRAY_T, "x"),                   # servo cable into the tray
            (ARM_D, -30.0, FOOT_H, "y"),                  # cartridge leads along the tower foot bar
            (ARM_D, -52.0, FOOT_H, "y")]


def cable_loop(axis):
    """Arch: two posts and a 2 mm bridge over a 5 wide, 6 tall channel along `axis`, standing on z = 0."""
    w, h, d = 5.0, 6.0, 4.0
    a = Box(w + 4, d, h + 2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    a -= Box(w, d + 1, h, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return a if axis == "x" else Rot(Z=90) * a


def cable_exits():
    """Fixed points where cables leave the frame (turntable frame, mm)."""
    x_near = -(SERVO_L - SHAFT_FROM_END) - FIT
    ax, ay = anchor_xy()
    return {"servo cable (open port end, between the rails)": (x_near - 6, 0.0, CHANNEL_Z / 2),
            "cartridge leads (anchor fin tie hole)": (float(ax), float(ay), ANCHOR_Z)}


def keepout():
    """Volumes the shell must not enter: {name: solid}, turntable frame."""
    from build123d import Solid, Wedge
    out = {}
    out["platter + record, 3 mm clear"] = Pos(0, 0, PLATTER_RIB_UNDERSIDE - 3) * Cylinder(
        PLAT_R + 3, Z_REC + 12 - (PLATTER_RIB_UNDERSIDE - 3), align=(Align.CENTER, Align.CENTER, Align.MIN))
    a0, a1 = sorted((arm_angle(REST_R), arm_angle(44.0)))
    a0, a1 = a0 - 6, a1 + 6
    reach = ARM_L + 30                                   # stylus + headshell + finger lift
    z0 = Z_ARM - CART_H - 3
    sweep = None
    for k in range(int(a1 - a0) + 1):                    # 1 deg slices: a thick fan around the pivot
        sl = Pos(ARM_D, 0, z0) * Rot(Z=a0 + k) * Pos(reach / 2, 0, 0) * Box(reach, 2 * math.radians(1.2) * reach + 40, 
                                                                         Z_ARM + 25 - z0, align=(Align.CENTER, Align.CENTER, Align.MIN))
        sweep = sl if sweep is None else sweep + sl
    out["arm sweep r 44 to rest r 130"] = sweep
    tail = STUB_L + 15
    cw = None
    for k in range(int(a1 - a0) + 1):
        sl = Pos(ARM_D, 0, Z_ARM - 30) * Rot(Z=a0 + k + 180) * Pos(tail / 2, 0, 0) * Box(tail, 40, 60,
                                                                                    align=(Align.CENTER, Align.CENTER, Align.MIN))
        cw = sl if cw is None else cw + sl
    out["counterweight swing"] = cw
    out["rest fork"] = rest_frame() * Pos(0, 0, REST_TOP - REST_WALL - 5) * Box(REST_LEN + 10, REST_W + 10, REST_WALL + 25,
                                                                              align=(Align.CENTER, Align.CENTER, Align.MIN))
    ax, ay = anchor_xy()
    out["cartridge lead loop"] = Pos(ARM_D, 0, ANCHOR_Z - 5) * Cylinder(math.hypot(ax - ARM_D, ay) + 15, Z_ARM - ANCHOR_Z + 10,
                                                                     align=(Align.CENTER, Align.CENTER, Align.MIN))
    (x0, y0), (x1, y1), h = ELECTRONICS_BAY
    out["electronics bay (reserved)"] = Pos((x0 + x1) / 2, (y0 + y1) / 2, 0) * Box(x1 - x0, y1 - y0, h,
                                                                                align=(Align.CENTER, Align.CENTER, Align.MIN))
    return out


def export_keepout_step(path):
    from build123d import Compound, export_step
    kids = []
    for name, shape in keepout().items():
        for i, so in enumerate(shape.solids()):
            so.label = name if i == 0 else f"{name} {i}"
            kids.append(so)
    export_step(Compound(children=kids, label="OpenVinyl shell keep-out"), str(path))


def peg_bottom_z():
    """Sleeve bottom = top of the T bar, on the shims actually used (SHIM_USE)."""
    return TIP_Z + (SHIM_USE - SHIM_NOM) - PIN_LEN + T_THICK


def cartridge():
    """The CKANDAY cartridge in the cartridge frame (stylus tip under the origin, +x rearward along the body,
    z = 0 the headshell underside): black housing, red stylus, cantilever and tip, four pins. Measured envelope;
    the snap window and recess on the top face are left out (the mount does not use them)."""
    top = -MOUNT_T
    bot = top - CART_T
    x0, x1 = -CART_FRONT, -CART_FRONT + CART_L
    body = Pos((x0 + x1) / 2, 0, bot) * Box(CART_L, CART_W, CART_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for y, z in ((-3.0, top - 3.0), (3.0, top - 3.0), (-2.0, top - 6.0), (2.0, top - 6.0)):    # L+ R+ / grounds
        body += Pos(x1 + CART_PIN_L / 2, y, z) * Rot(Y=90) * Cylinder(0.6, CART_PIN_L)
    body += Pos(1.5, 0, -CART_H) * Rot(Y=-35) * Box(4.0, 0.6, 0.6, align=(Align.MIN, Align.CENTER, Align.MIN))  # cantilever
    body += Pos(0, 0, -CART_H) * Cylinder(0.3, CART_TIP_DROP + 0.6, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return body


def cartridge_stylus():
    """The removable red stylus alone (its envelope on the underside front), for the pull-off check."""
    top = -MOUNT_T
    return Pos(-CART_FRONT + CART_RED / 2, 0, top - CART_T) * Box(CART_RED, CART_W, CART_T / 2,
                                                                   align=(Align.CENTER, Align.CENTER, Align.MIN))


def cart_mount():
    """Printed mount, cartridge frame: a 3.5 mm plate against the cartridge's top face, two M2 holes on the
    1/2 in line that the headshell screws self-tap into from above (M2x8 button heads: 3.5 mm of thread, no nuts),
    and two cheeks on the black part; the cheek on +y has a boss for a third M2x8 that clamps the body sideways.
    Prints headshell face down, cheeks up: no overhangs."""
    x0, x1 = 1.0, -CART_FRONT + CART_L - 0.5
    hw = CART_W / 2
    m = Pos((x0 + x1) / 2, 0, -MOUNT_T) * Box(x1 - x0, CART_W + 2 * CHEEK_T + CLAMP_GAP + 0.3, MOUNT_T,
                                              align=(Align.CENTER, Align.CENTER, Align.MIN))
    m += Pos(MOUNT_SCREW_X, 0, -MOUNT_T) * Box(7.0, MOUNT + 5.0, MOUNT_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    for y in (-MOUNT / 2, MOUNT / 2):
        m -= Pos(MOUNT_SCREW_X, y, -MOUNT_T) * Cylinder(MOUNT_PILOT / 2, MOUNT_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    c0, c1 = CHEEK_X
    zc = -MOUNT_T - CHEEK_H
    yl0 = -hw - 0.15 - CHEEK_T                                       # snug cheek, -y
    m += Pos((c0 + c1) / 2, yl0 + CHEEK_T / 2, zc) * Box(c1 - c0, CHEEK_T, CHEEK_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    yr0 = hw + CLAMP_GAP                                             # clamp cheek, +y, with its screw boss
    m += Pos((c0 + c1) / 2, yr0 + CHEEK_T / 2, zc) * Box(c1 - c0, CHEEK_T, CHEEK_H, align=(Align.CENTER, Align.CENTER, Align.MIN))
    bx, bz = (c0 + c1) / 2, -MOUNT_T - CHEEK_H / 2
    m += Pos(bx, yr0 + CHEEK_T + 1.5, zc) * Box(6.0, 3.0, CHEEK_H + MOUNT_T, align=(Align.CENTER, Align.CENTER, Align.MIN))
    m -= Pos(bx, yr0 + (CHEEK_T + 3.0) / 2, bz) * Rot(X=90) * Cylinder(MOUNT_PILOT / 2, CHEEK_T + 3.0 + 0.2)
    return m


def clamp_screw(engaged=True):
    """M2x8 button-head clamp screw stand-in (head outside the boss), tip on the body's side when engaged."""
    c0, c1 = CHEEK_X
    hw = CART_W / 2
    tip = hw if engaged else hw + CLAMP_GAP
    bx, bz = (c0 + c1) / 2, -MOUNT_T - CHEEK_H / 2
    s = Pos(bx, tip + 4.0, bz) * Rot(X=90) * Cylinder(1.0, 8.0)
    return s + Pos(bx, tip + 8.0 + 0.8, bz) * Rot(X=90) * Cylinder(1.75, 1.6)


clip_standin = cart_mount                # older scripts (briefcase.py, schematics) call it by its old name


def pivot_screw():
    """M3x8 ISO 7380 button head in the arm frame, as fitted: head pulled up to the counterbore top, face down at
    PIVOT_FACE, 2.0 mm hex socket, shank up (pokes 3.4 mm out of the hub top)."""
    s = Pos(0, 0, PIVOT_FACE) * Cylinder(PIVOT_HEAD_D / 2, PIVOT_HEAD_K, align=(Align.CENTER, Align.CENTER, Align.MIN))
    s -= Pos(0, 0, PIVOT_FACE) * Cylinder(1.15, PIVOT_SOCKET_T, align=(Align.CENTER, Align.CENTER, Align.MIN))  # socket
    s += Pos(0, 0, PIVOT_FACE + PIVOT_HEAD_K) * Cylinder(1.5, PIVOT_LEN, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return s


SERVO_STEP = Path(__file__).parent / "vendor" / "STS3215_03a.step"   # SO-ARM100 repo, STEP/SO100 (Nov 21)
STEP_AXIS_X, STEP_COVER_Z = 12.5, -17.7            # horn axis and cover plate in the STEP's own frame


def servo():
    """STS3215 from the SO-ARM100 STEP (body and horn in one solid), horn axis on x = 0, cover plate at SEAT.
    Its horn centre is 10.2 mm from the horn end; the real servo measures 9.5 (measured Nov 21), which the
    pocket uses, so the model overlaps the horn-end wall by 0.3 mm in the assembly."""
    from build123d import import_step
    return Pos(-STEP_AXIS_X, 0, SEAT - STEP_COVER_Z) * import_step(str(SERVO_STEP))


def horn():
    return Pos(0, 0, Z_HORN - 3) * Cylinder(HORN_OD / 2, 3, align=(Align.CENTER, Align.CENTER, Align.MIN))


def record():
    r = Cylinder(R_REC, 1.2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    r -= Cylinder(7.24 / 2, 1.2, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return Pos(0, 0, Z_HORN + HUB_H) * r


def assembly(r_stylus=65.0, platter_deg=0.0):
    """{name: (shape in the turntable frame, group)}; groups: frame, platter, arm."""
    sgn = headshell_sign()
    turn = Rot(Z=platter_deg)
    A = Pos(ARM_D, 0, Z_ARM) * Rot(Z=arm_angle(r_stylus))
    shell_frame = Pos(ARM_L, 0, 0) * Rot(Z=180 + sgn * ARM_BETA)
    return {
        "base": (base(), "frame"),
        "servo STS3215": (servo(), "frame"),
        "peg": (Pos(ARM_D, 0, peg_bottom_z()) * peg(), "frame"),
        "T-pin": (Pos(ARM_D, 0, TIP_Z + PIVOT_RISE - PIN_LEN) * tpin(), "frame"),
        "shims": (Pos(ARM_D, 0, FLOOR_Z) * shim(SHIM_USE), "frame"),
        "horn": (turn * horn(), "platter"),
        "platter": (turn * Pos(0, 0, Z_HORN) * platter(), "platter"),
        "spindle": (turn * Pos(0, 0, Z_HORN + HUB_H) * spindle(), "platter"),
        "record 7 in": (turn * record(), "platter"),
        "arm": (A * arm_side(arm()), "arm"),
        "pivot screw": (A * pivot_screw(), "arm"),
        "weight": (A * arm_side(Pos(-60, 0, -(30 - 2 - BEAM - 0.5) - 0.3) * weight()), "arm"),
        "cartridge (CKANDAY)": (A * arm_side(shell_frame * cartridge()), "arm"),
        "headshell": (A * arm_side(shell_frame * cart_mount()), "arm"),
    }


def export_assembly_step(path, r_stylus=65.0):
    from build123d import Compound, export_step
    kids = []
    for name, (shape, _) in assembly(r_stylus).items():
        solids = shape.solids()
        assert len(solids) == 1, (name, len(solids))
        solid = solids[0]                  # a bare Solid, so Onshape imports a part, not a sub-assembly
        solid.label = name
        kids.append(solid)
    export_step(Compound(children=kids, label="OpenVinyl turntable"), str(path))


def main():
    OUT.mkdir(exist_ok=True)
    parts = {
        "platter": Rot(X=180) * platter(),         # top face on the bed
        "spindle": spindle(),                       # key on the bed, column up: no overhang (Nov 22)
        "base": base(),
        "peg": peg(),                              # upright: the pin bore prints round
        "arm": arm_side(arm()),
        "weight": Rot(X=180) * arm_side(weight()),  # carriage roof on the bed
    }
    parts.update({f"shim_{t:.0f}": shim(t) for t in SHIMS})   # flat
    parts["cart_mount"] = Rot(X=180) * arm_side(cart_mount())    # headshell face on the bed, cheeks up
    for name, shape in parts.items():
        export_stl(shape, str(OUT / f"{name}.stl"))
        bb = shape.bounding_box()
        print(f"{name:8s} {bb.size.X:6.1f} x {bb.size.Y:6.1f} x {bb.size.Z:5.1f} mm  volume {shape.volume / 1000:6.1f} cm3")


if __name__ == "__main__":
    main()
