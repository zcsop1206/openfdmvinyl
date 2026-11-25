"""
spine_assembly.py - the spine build (spine.py) as an assembly, for the clash check and the Onshape upload.

    python spine_assembly.py      # clash check over the sweep, writes spine_assembly.step

spine.py overrides turntable.py's arm constants on import (135 mm arm, pivot 119.8 mm, 45 mm stub); this module
imports it first, then swaps the full base for the spine and the solid platter for the open one. The carriage
sits where balance.py puts it for 5 g with the spine arm: x -34 (7.1 g of coins).
"""
from pathlib import Path

from build123d import Pos, Rot

import spine as S                         # noqa: F401  (overrides T's arm constants first)
import turntable as T

HERE = Path(__file__).parent
CARRIAGE_X = -34.0                        # balance.py with spine.py's overrides, 5 g force
GROUPS = {"frame": ["spine", "servo STS3215", "peg", "T-pin", "shims", "damper cup", "clamp screw"],
          "platter": ["horn", "platter (open)", "spindle", "record 7 in", "clamp ring", "centre weight"],
          "arm": ["arm", "weight", "cartridge (CKANDAY)", "headshell", "pivot screw", "damper skirt"]}


def assembly(r_stylus=65.0, platter_deg=0.0):
    a = T.assembly(r_stylus, platter_deg)
    out = {"spine": (S.spine(), "frame")}
    out.update({k: v for k, v in a.items() if k not in ("base", "platter", "weight")})
    out["platter (open)"] = (Rot(Z=platter_deg) * Pos(0, 0, T.Z_HORN) * S.open_platter(), "platter")
    A = Pos(T.ARM_D, 0, T.Z_ARM) * Rot(Z=T.arm_angle(r_stylus))
    out["weight"] = (A * T.arm_side(Pos(CARRIAGE_X, 0, -(30 - 2 - T.BEAM - 0.5) - 0.3) * T.weight()), "arm")
    # Nov 24: damper, clamp screw, clamp ring (steady.py, rim_clamp.py)
    out["damper cup"] = (Pos(T.ARM_D, 0, 0) * S.SD.cup(), "frame")
    out["clamp screw"] = (Pos(T.ARM_D, 0, 0) * S.SD.clamp_screw(), "frame")
    out["damper skirt"] = (A * T.arm_side(S.SD.skirt()), "arm")
    t_rec = T.Z_REC - T.Z_HORN - T.HUB_H
    out["clamp ring"] = (Rot(Z=platter_deg) * Pos(0, 0, T.Z_HORN + S.RC.ring_top_z(t_rec)) * Rot(Z=-S.RC.snug_turn(t_rec))
                         * S.RC.ring(), "platter")
    out["centre weight"] = (Rot(Z=platter_deg) * Pos(0, 0, T.Z_REC) * S.RC.centre_weight(), "platter")
    return out


def export_step(path, r_stylus=65.0):
    from build123d import Compound, export_step as _export
    kids = []
    for name, (shape, _) in assembly(r_stylus).items():
        solids = shape.solids()
        assert len(solids) == 1, (name, len(solids))
        solid = solids[0]
        solid.label = name
        kids.append(solid)
    _export(Compound(children=kids, label="OpenVinyl spine build"), str(path))


def check():
    from motion import dist
    moving = ("arm", "weight", "cartridge (CKANDAY)", "headshell", "pivot screw", "damper skirt")
    fixed = ("spine", "servo STS3215", "peg", "platter (open)", "record 7 in", "spindle", "horn", "damper cup",
             "clamp screw", "clamp ring", "centre weight")
    worst = {}
    for r in (130, 110, 95, 84, 75, 65, 55, 46, 44):
        a = assembly(r_stylus=r)
        for m in moving:
            for f in fixed:
                if (m, f) in (("pivot screw", "peg"), ("damper skirt", "peg")):
                    continue
                d = dist(a[m][0], a[f][0])
                if (m, f) not in worst or d < worst[(m, f)][0]:
                    worst[(m, f)] = (d, r)
    print(f"arm {T.ARM_L} mm, pivot {T.ARM_D} mm, offset {T.ARM_BETA:.2f} deg, stub {T.STUB_L} mm, carriage x {CARRIAGE_X}")
    bad = 0
    for (m, f), (d, r) in sorted(worst.items(), key=lambda kv: kv[1][0]):
        flag = "  <-- CLASH" if d <= 0.05 else ("  <-- tight" if d < 2 else "")
        bad += d <= 0.05 and not (m == "cartridge (CKANDAY)" and f == "record 7 in")
        print(f"  {m:28s} | {f:16s} {d:6.2f} at r {r}{flag}")
    return bad


def mount_check():
    """The cartridge in its headshell and the headshell on the arm's pad (headshell.py)."""
    import headshell as HS
    return len(HS.fit_checks())


if __name__ == "__main__":
    mount_check()
    check()
    export_step(HERE / "spine_assembly.step")
    print("wrote spine_assembly.step")
