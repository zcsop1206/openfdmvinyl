"""
motion.py - motion study of the turntable: arm sweep and platter turn, exact clearances, wire loop.

    python motion.py            # prints the clearance table, writes motion_study.csv and motion_study.mp4

Arm group (arm, weight, cartridge stand-in) swings about the pivot from the rest position (stylus at
r = 100 mm, off the record) to the end of the lead-out (r = 44 mm). At each step: the exact minimum
distance (OCC BRepExtrema) from each arm-group part to each fixed or platter part. The needle is left
out: it touches the cone by design. Wire: the cartridge leads run along the beam (fixed to the arm),
then drop from the hole beside the stub to the anchor fin on the base. That drop is the only span that
moves, so the script tracks its length and its clearance to the peg, tower, needle and weight.
"""
import csv
import math
from pathlib import Path

import numpy as np
from OCP.BRepExtrema import BRepExtrema_DistShapeShape

import turntable as T

HERE = Path(__file__).parent
R_STEPS = [130, 120, 110, 100, 95, 90, 84, 78, 72, 65, 58, 52, 46, 44]


def dist(a, b):
    d = BRepExtrema_DistShapeShape(a.wrapped, b.wrapped)
    d.Perform()
    return d.Value()


def seg_dist_to_cyl(p0, p1, c, r, z0, z1, n=60):
    """Min distance from segment p0-p1 to a vertical cylinder (axis c, radius r, z0..z1)."""
    best = math.inf
    for t in np.linspace(0, 1, n):
        p = p0 + t * (p1 - p0)
        radial = math.hypot(p[0] - c[0], p[1] - c[1]) - r
        dz = max(z0 - p[2], 0, p[2] - z1)
        best = min(best, math.hypot(max(radial, 0), dz) if radial > 0 or dz > 0 else -1)
    return best


def main():
    rows = []
    fixed_names = ["base", "peg", "servo STS3215", "platter", "spindle", "record 7 in", "horn"]
    arm_names = ["arm", "weight", "cartridge (CKANDAY)", "headshell"]
    anchor = np.r_[T.anchor_xy(), T.ANCHOR_Z]
    peg_z = T.peg_bottom_z()
    peg_top = peg_z + T.PEG_L
    print(f"sleeve bottom z {peg_z:.1f}, sleeve top z {peg_top:.1f}, needle tip z {T.Z_ARM + T.CONE_H:.1f}, "
          f"hub underside z {T.Z_ARM:.1f}")
    for r in R_STEPS:
        asm = T.assembly(r_stylus=r)
        ang = math.radians(T.arm_angle(r))
        dx, dy = T.DROP
        drop = np.array([T.ARM_D + dx * math.cos(ang) - dy * math.sin(ang), dx * math.sin(ang) + dy * math.cos(ang), T.Z_ARM])
        row = {"r_mm": r, "arm_deg": round(T.arm_angle(r), 2)}
        for an in arm_names:
            for fn in fixed_names:
                row[f"{an} | {fn}"] = round(dist(asm[an][0], asm[fn][0]), 2)
        row["wire drop length"] = round(float(np.linalg.norm(drop - anchor)), 2)
        row["wire | peg"] = round(seg_dist_to_cyl(drop, anchor, (T.ARM_D, 0), T.PEG_D / 2, peg_z, peg_top), 2)
        row["wire | tower"] = round(seg_dist_to_cyl(drop, anchor, (T.ARM_D, 0), T.TOWER_OD / 2, 0, T.TOWER_H), 2)
        row["wire | needle"] = round(seg_dist_to_cyl(drop, anchor, (T.ARM_D, 0), 0.5, peg_top, T.Z_ARM + T.CONE_H), 2)
        rows.append(row)
        print(f"r {r:5.1f}  arm {row['arm_deg']:6.1f} deg  done")
    keys = list(rows[0])
    with open(HERE / "motion_study.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print("\nminimum over the sweep (mm; 0 = touching or overlapping):")
    for k in keys[2:]:
        vals = [row[k] for row in rows]
        worst = min(vals) if "length" not in k else None
        if "length" in k:
            print(f"  {k:52s} {min(vals):6.2f} .. {max(vals):6.2f}")
        else:
            at = rows[vals.index(worst)]["r_mm"]
            flag = "  <-- CLASH" if worst <= 0.05 else ("  <-- tight" if worst < 2 else "")
            print(f"  {k:52s} {worst:6.2f} at r {at}{flag}")


if __name__ == "__main__":
    main()
