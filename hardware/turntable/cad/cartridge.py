"""
cartridge.py - the CKANDAY "universal" ceramic cartridge (a Chuo Denshi CZ-800 clone) as measured, for designing
the headshell around it.

    python cartridge.py        # writes cartridge.step and cartridge_views.png (same views as the photos)

Model frame: x along the body from the red front (x = 0) back to the rear face (x = LEN), y across (centred),
z up from the bottom face (z = 0), stylus down. Measured with calipers, Nov 22 2026; the photos with the
lettered features are in ../docs_cartridge/. Values marked ASSUMED are read from photos or guessed and wait for a
measurement; nothing a fit depends on should rest on them without a check.
"""
from pathlib import Path

from build123d import Align, Box, Compound, Cylinder, Plane, Polygon, Pos, Rot, extrude, export_step

HERE = Path(__file__).parent
MIN = (Align.CENTER, Align.CENTER, Align.MIN)

# Sources: calipers (Nov 22) and two straight-on photos with the caliper scale in frame (top and side, Nov 22),
# read on a pixel grid and scaled by the measured width/height; the two agree within about 0.3 mm.
# ---- envelope -----------------------------------------------------------------------------------------------
LEN = 27.0           # red front to the lower rear face, where the pins come out (30.8 to the pin tips - 3.8)
W = 9.5              # width, black and red
H_REAR = 9.4         # top of the raised rear block (the body's tallest point)
H_MID = 8.5          # C: top of the rails along the middle section
BLACK_X0 = 0.6       # photo: the black nose starts 0.6 behind the red front
OVERHANG_X = 28.5    # photo: the upper rear block overhangs the lower rear face (and the pins) to here
OVERHANG_Z = 5.9     # photo +-0.3: underside of that overhang (the upper pins, top at 5.8, exit just under it)
# ---- top face ------------------------------------------------------------------------------------------------
REAR_BLOCK = 6.3     # B: raised rear block length
STEP_X = OVERHANG_X - REAR_BLOCK  # 22.2: its front, a short ramp up from the rails
STEP_RAMP = 0.8      # photo: the step rises over about 0.8 mm
WIN_L, WIN_W = 3.5, 4.5           # G: window in the rear block (along x, across y)
WIN_BACK_WALL = 1.1  # G: back edge 1.1 in from the rear block's back face (calipers; the photo read 0.5)
WIN_X0 = OVERHANG_X - WIN_BACK_WALL - WIN_L     # 23.9
WIN_DEPTH = 2.0      # more than 1.5 (a central block inside stops the caliper); the holder does not use the window
WIN_BLOCK_W = 1.2    # ASSUMED: that central block's width (photo), rising to 0.5 below the block top
REC_L, REC_W, REC_D = 13.2, 6.1, 1.3     # I: recess in the middle section, ending at the step; rails J 1.6-1.7
FRONT_H = 6.4        # photo (D measured 5.3, marked unsure): black nose height at its front
FRONT_SLOPE_TO = 4.3  # photo: the nose slope reaches H_MID here
NOTCH_W, NOTCH_L = 4.8, 3.4       # H: notch open at the front between two black arms (the stylus grip)
# ---- red stylus ----------------------------------------------------------------------------------------------
RED_H = 4.6          # E (measured 4.5, unsure; photo 4.7 at the front): top edge of the red on the side
RED_TOP_L = 18.0     # R1: red along its top edge (side), from the red front
RED_BOT_L = 15.8     # R2: red along the bottom, from the red front
RED_FRONT_LIFT = 1.6  # photo: the red underside slopes up to 1.6 above the bottom plane at the front
TIP_X = 2.0          # stylus tip behind the red front (photo 2.2)
TIP_BELOW_TOP = 9.9  # measured from the raised rear block's top (uncertain; the photo reads about 10.5)
TIP_Z = H_REAR - TIP_BELOW_TOP    # -0.5: tip below the bottom plane
# ---- rear ----------------------------------------------------------------------------------------------------
TAB_W, TAB_T, TAB_OUT = 2.8, 1.2, 1.0   # F: tab at the top of the overhang's back face (photo: 1.0 out)
PIN_D, PIN_L = 1.2, 3.8
PINS = ((5.2, 6.5), (2.5, 3.5))         # M: (row height above the bottom, centre-to-centre spacing)
SLOT_L, SLOT_W, SLOT_FROM_REAR, SLOT_D = 5.3, 0.7, 1.1, 1.0   # L: slot in the black bottom; depth ASSUMED

MASS_G = 2.1         # published CZ-800 mass (not weighed)


def prism_xz(pts, y0, y1):
    """An (x, z) profile extruded across y from y0 to y1 (counter-clockwise first)."""
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    pts = pts if area > 0 else pts[::-1]
    # Plane.XZ: local x = global x, local y = global z, normal -y; extrude from y1 toward y0
    return Pos(0, y1, 0) * extrude(Plane.XZ * Polygon(*pts, align=None), y1 - y0)


def box(x0, x1, y0, y1, z0, z1):
    return Pos((x0 + x1) / 2, (y0 + y1) / 2, z0) * Box(x1 - x0, y1 - y0, z1 - z0, align=MIN)


def red():
    """The removable stylus: lower front of the body, slanted back edge, underside rising to the front, plus the
    cantilever and tip. It slides off forward."""
    prof = [(0.0, RED_FRONT_LIFT), (RED_BOT_L, 0.0), (RED_TOP_L, RED_H), (0.0, RED_H)]
    r = prism_xz(prof, -W / 2, W / 2)
    r += Pos(TIP_X, 0, TIP_Z) * Cylinder(0.3, -TIP_Z + 0.9, align=MIN)              # tip and shank
    r += Pos(TIP_X, 0, TIP_Z + 0.3) * Rot(Y=-15) * Box(9.0, 0.6, 0.6, align=(Align.MIN, Align.CENTER, Align.CENTER))  # cantilever, back to the red's underside (photo)
    return r


def black():
    """The housing: everything above and behind the red: sloped nose with the front notch, rails and recess,
    the raised rear block overhanging the pins, its window and tab, the pins and the bottom slot."""
    prof = [(BLACK_X0, RED_H), (BLACK_X0, FRONT_H), (FRONT_SLOPE_TO, H_MID), (STEP_X - STEP_RAMP, H_MID),
            (STEP_X, H_REAR), (OVERHANG_X, H_REAR), (OVERHANG_X, OVERHANG_Z), (LEN, OVERHANG_Z), (LEN, 0.0),
            (RED_BOT_L, 0.0), (RED_TOP_L, RED_H)]
    b = prism_xz(prof, -W / 2, W / 2)
    b -= box(WIN_X0, WIN_X0 + WIN_L, -WIN_W / 2, WIN_W / 2, H_REAR - WIN_DEPTH, H_REAR + 1)        # G
    b += box(WIN_X0, WIN_X0 + WIN_L, -WIN_BLOCK_W / 2, WIN_BLOCK_W / 2, H_REAR - WIN_DEPTH, H_REAR - 0.5)  # block in G
    b -= box(STEP_X - STEP_RAMP - REC_L, STEP_X - STEP_RAMP, -REC_W / 2, REC_W / 2, H_MID - REC_D, H_MID + 1)  # I
    b -= box(-1, NOTCH_L, -NOTCH_W / 2, NOTCH_W / 2, RED_H, H_MID + 1)                             # H
    b -= box(LEN - SLOT_FROM_REAR - SLOT_L, LEN - SLOT_FROM_REAR, -SLOT_W / 2, SLOT_W / 2, -1, SLOT_D)  # L
    b += box(OVERHANG_X, OVERHANG_X + TAB_OUT, -TAB_W / 2, TAB_W / 2, H_REAR - TAB_T, H_REAR)       # F
    for z, c in PINS:
        for s in (-1, 1):
            b += Pos(LEN + PIN_L / 2, s * c / 2, z) * Rot(Y=90) * Cylinder(PIN_D / 2, PIN_L)
    return b


def cartridge():
    return {"black": black(), "red": red()}


def check():
    """Sanity: the measured parts add up."""
    rails = (W - REC_W) / 2
    print(f"CKANDAY model: {LEN} x {W} x {H_REAR} mm (pins +{PIN_L}); rails {rails:.2f} (measured 1.6); "
          f"step {H_REAR - H_MID:.1f} mm at x {STEP_X}; recess x {STEP_X - STEP_RAMP - REC_L:.1f}..{STEP_X - STEP_RAMP:.1f}; "
          f"window x {WIN_X0}..{WIN_X0 + WIN_L}; overhang to {OVERHANG_X}, tab to {OVERHANG_X + TAB_OUT}, pin tips "
          f"{LEN + PIN_L}; tip at x {TIP_X}, z {TIP_Z:.1f}")
    parts = cartridge()
    for n, p in parts.items():
        print(f"  {n:6s} {len(p.solids())} solid(s), {p.volume:7.1f} mm3")


def views(path):
    """Shaded views matching the photos: side (stylus down, front left), top from the rear-left, bottom, rear."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    parts = cartridge()
    items = [(parts["black"], "#2b2b29"), (parts["red"], "#c0392b")]
    light = np.array([0.3, -0.6, 0.75]); light /= np.linalg.norm(light)
    fig = plt.figure(figsize=(12, 9.5), dpi=110)
    for k, (el, az, title) in enumerate([(0, -90, "side (1_side)"), (40, -125, "top, from the front-left (2_top)"),
                                         (-90, -90, "bottom (3_bottom)"), (25, 20, "rear (4_rear)")]):
        ax = fig.add_subplot(2, 2, k + 1, projection="3d")
        for shp, col in items:
            v, f = shp.tessellate(0.02, 0.15)
            t = np.array([tuple(q) for q in v])[np.array(f)]
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1)[:, None] + 1e-12
            sh = 0.35 + 0.65 * np.abs(n @ light)
            c = np.array(matplotlib.colors.to_rgb(col))[None, :] * sh[:, None]
            ax.add_collection3d(Poly3DCollection(t, facecolors=np.clip(c * 1.6, 0, 1), linewidths=0.15, edgecolors="#00000030"))
        ax.set_xlim(0, 31); ax.set_ylim(-15.5, 15.5); ax.set_zlim(-11, 20); ax.set_box_aspect((1, 1, 1)); ax.dist = 6.5
        ax.view_init(el, az); ax.set_axis_off(); ax.set_title(title, fontsize=9)
    fig.suptitle("CKANDAY cartridge model from the caliper measurements (front at x = 0, pins at the rear)", fontsize=10)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight")


if __name__ == "__main__":
    check()
    p = cartridge()
    for n, s in p.items():
        s.label = n
    export_step(Compound(children=list(p.values()), label="CKANDAY cartridge"), str(HERE / "cartridge.step"))
    views(HERE / "cartridge_views.png")
    print("wrote cartridge.step and cartridge_views.png")
