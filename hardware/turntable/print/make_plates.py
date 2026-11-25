"""
make_plates.py - slice the turntable parts into Bambu Studio projects for the P1S (0.4 mm nozzle, PLA).

    python make_plates.py            # writes plate_*.3mf here, sliced, ready to open and send
    python make_plates.py spine      # the spine build (../cad/stl_spine) into spine/plate_*.3mf, same settings
    python make_plates.py spine 2 8  # only those spine plates (by number)
    python make_plates.py briefcase  # the briefcase top half (../cad/stl_briefcase) into briefcase/plate_*.3mf

One project per plate, because the plates want different walls and infill. The STLs in ../cad/stl are
already in print orientation, so the slicer only arranges them (no auto-orient, no supports). Profiles
are Bambu Studio's own system presets (0.20mm Standard, Bambu PLA Basic, P1S 0.4) flattened from their
inherit chains, with the overrides below. The record disc is not here: it is the Bambu base disc with
the groove spliced in (see docs/BUILD.md).
"""
import json
import re
import subprocess
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).parent
STL = HERE.parent / "cad" / "stl"
STUDIO = Path(r"C:\Program Files\Bambu Studio")
PROFILES = STUDIO / "resources" / "profiles" / "BBL"
MACHINE, PROCESS, FILAMENT = "Bambu Lab P1S 0.4 nozzle", "0.20mm Standard @BBL X1C", "Bambu PLA Basic @BBL P1S 0.4 nozzle"
BED = "Textured PEI Plate"    # the P1S used here (Nov 22; the presets default to the Cool Plate, a 35 C bed)

PLATES = {
    # name: (parts, process overrides, why)
    "1_base": (["base"], {"wall_loops": "3", "sparse_infill_density": "15%", "sparse_infill_pattern": "gyroid"},
               "Prints as assembled. 3 walls carry the tower and rest-post roots; the strips are walls only."),
    "2_platter": (["platter"], {"wall_loops": "3", "sparse_infill_density": "15%", "sparse_infill_pattern": "gyroid",
                                "top_shell_layers": "5", "bottom_shell_layers": "5"},
                  "Top face down, so the record seat is the bed surface. 5 shells keep the 3 mm plate solid."),
    "3_arm_small": (["arm", "weight", "peg", "spindle", "shim_1", "shim_2", "shim_4", "headshell"],
                    {"wall_loops": "4", "sparse_infill_density": "40%", "sparse_infill_pattern": "gyroid"},
                    "Stiffness: the 8 mm beam is nearly all wall; the sleeve, spindle, shims and the cartridge mount print nearly solid."),
}
# spine build (cad/spine.py): the same settings as the full build's base, platter and arm plates
SPINE_PLATES = {
    "1_spine": (["spine"], PLATES["1_base"][1], "Prints as assembled. 3 walls make the hollow tower's 2.4 mm wall nearly solid."),
    "2_platter_open": (["platter_open"], PLATES["2_platter"][1], "Top face down; 5 shells keep the rim and spokes solid."),
    "3_arm_small": PLATES["3_arm_small"],
    "4_sleeve": (["peg"], PLATES["3_arm_small"][1], "Pivot sleeve alone (Nov 24): V-bottomed pin bore and the clamp screw's slot."),
    "5_fit_reprint": (["headshell", "weight", "peg"], PLATES["3_arm_small"][1],
                      "Headshell, acorn-nut carriage and sleeve (Nov 24): the arm's small parts without the arm."),
    "6_spindles": ([f"spindle_{d}" for d in (685, 695, 705, 715, 725)], PLATES["3_arm_small"][1],
                   "Keyed spindle fit set (Nov 22): the key fills the hub's socket and screw slots; 6.85-7.25 mm above "
                   "the platter, 1-5 dimples. Keep the one that slides snugly into the disc's hole."),
    "7_headshell": (["headshell"], PLATES["3_arm_small"][1],
                    "Headshell alone (Nov 23): the pocket pitched 5.8 deg so the red stylus's underside lies flat and the "
                    "body clears the record by 1.9 mm. Same arm pad, bolts and dowel (16 mm now)."),
    "8_clamp_damper": (["clamp_ring", "centre_weight", "damper_cup", "damper_skirt"], PLATES["3_arm_small"][1],
                       "Nov 24: twist-lock clamp ring for the disc edge (flange down), centre weight for 18 acorn nuts, "
                       "damper cup (floor down) and hub skirt (washer down)."),
    "9_fit_test": (["fit_tower", "fit_sleeve", "fit_servo", "fit_hub", "fit_stub", "fit_carriage", "fit_carriage_nuts", "fit_rim", "fit_ring",
                    "fit_weight", "damper_cup", "damper_skirt"], PLATES["3_arm_small"][1],
                   "Nov 24: fit-test coupons cut from the real parts (cad/fit_test.py). Print before plates 1-3 and 8."),
}

# briefcase top half (cad/briefcase_print.py, Nov 24): the top half first, one piece, face-down with tree
# supports; low infill on the big non-load-bearing panels; black PLA, silk silver for the accents
SILK = "Bambu PLA Silk+ @BBL P1S 0.4 nozzle"
LID = {"wall_loops": "3", "sparse_infill_density": "5%", "sparse_infill_pattern": "gyroid", "top_shell_layers": "5",
       "bottom_shell_layers": "4", "enable_support": "1", "support_type": "tree(auto)",
       "support_on_build_plate_only": "1", "brim_type": "no_brim", "_full_bed": True}
BRIEFCASE_PLATES = {
    "b1_top_half": (["top_half"], LID, "Lid, face-down: 5 % gyroid, tree supports from the bed only (the arm bays and the dial "
                                      "pocket; the well floor is the pan, b5), no brim (252 mm on the 256 mm bed)."),
    "b2_deck": (["deck"], {"wall_loops": "2", "sparse_infill_density": "10%", "brim_type": "no_brim", "_full_bed": True},
                "Deck plate, face-down (the speed marks on the bed); 1.5 mm, so it prints solid."),
    "b3_feet_cap": (["foot_0", "foot_1", "foot_2", "foot_3", "cup_cap", "cup_plug"],
                    {"wall_loops": "2", "sparse_infill_density": "5%", "sparse_infill_pattern": "gyroid"},
                    "Four hollow corner feet for playing the lid alone, and the damper cup's cap and plug for carrying."),
    "b4_silver": (["dial", "rest"], {"wall_loops": "3", "sparse_infill_density": "15%", "_filament": SILK},
                  "Silk silver: speed dial and arm rest."),
    "b5_pan": (["well_pan"], {"wall_loops": "3", "sparse_infill_density": "5%", "sparse_infill_pattern": "gyroid",
                              "enable_support": "0"},
               "Well pan (the platter-well floor and the servo pocket), top face on the bed: no supports. Three M3x12 "
               "hold it on the lid's cone ring."),
}


def index(kind):
    out = {}
    for f in (PROFILES / kind).glob("*.json"):
        try:
            out[json.loads(f.read_text(encoding="utf-8"))["name"]] = f
        except (KeyError, json.JSONDecodeError):
            pass
    return out


def flatten(name, kind, idx):
    d = json.loads(idx[name].read_text(encoding="utf-8"))
    if "inherits" in d:
        parent = flatten(d.pop("inherits"), kind, idx)
        parent.update(d)
        d = parent
    return d


def write_profiles(out, filament=None):
    out.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, name in (("machine", MACHINE), ("process", PROCESS), ("filament", filament or FILAMENT)):
        d = flatten(name, kind, index(kind))
        d.pop("inherits", None)
        d["from"] = "system"
        paths[kind] = out / f"{kind}.json"
        paths[kind].write_text(json.dumps(d, indent=1), encoding="utf-8")
    return paths


def summary(threemf):
    with zipfile.ZipFile(threemf) as z:
        info = z.read("Metadata/slice_info.config").decode()
    t = re.search(r'key="prediction" value="(\d+)"', info)
    g = re.findall(r'used_g="([\d.]+)"', info)
    return (int(t.group(1)) if t else None), sum(map(float, g))


LOAD_LINE_LOOP = ("G0 Y11 E0.700", "G0 X239.5", "G0 E0.2", "G0 Y1.5 E0.700", "G0 X18 E15")


def full_bed_machine(base_machine, out):
    """Machine profile for a part that covers the bed (the 252 mm briefcase lid, Nov 24): the excluded front-left
    corner is cleared, and the start G-code's nozzle load line is one pass along the front edge at Y 0.8 instead of a
    loop up to Y 11 and back, so it stays in front of the part (face-down, the lid's 3 mm rounded edge puts its first
    layer 5 mm in from the bed's edge)."""
    d = json.loads(base_machine.read_text(encoding="utf-8"))
    d["bed_exclude_area"] = ["0x0"]
    g = d["machine_start_gcode"]
    g = g if isinstance(g, str) else "".join(g)
    lines = g.split(chr(10))
    out_lines, n = [], 0
    for ln in lines:
        if ln.startswith("G1 X18.0 Y1.0 Z0.8"):
            ln = ln.replace("Y1.0", "Y0.8") + " ; OpenVinyl full-bed plate: load line kept at the front edge"
        if any(ln.startswith(k) for k in LOAD_LINE_LOOP) and "nozzle load" not in ln:
            n += 1
            continue
        out_lines.append(ln)
    assert n == len(LOAD_LINE_LOOP), ("start G-code changed; check the load line", n)
    d["machine_start_gcode"] = chr(10).join(out_lines)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return out


def centred_copies(stl, parts, out_dir):
    """STL copies moved to the bed's centre (128, 128), for slicing with --arrange 0."""
    import trimesh
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for q in parts:
        m = trimesh.load(str(stl / f"{q}.stl"))
        lo, hi = m.bounds
        m.apply_translation([128 - (lo[0] + hi[0]) / 2, 128 - (lo[1] + hi[1]) / 2, -lo[2]])
        path = out_dir / f"{q}.stl"
        m.export(str(path))
        paths.append(path)
    return paths


def slice_one(plate, parts, over, stl, dest, work, base, out_name=None):
    """Slice one plate into dest; returns (seconds, grams). Keys starting with _ are options, not process settings:
    _filament (a filament preset), _full_bed (see full_bed_machine; one part, centred, no arrange)."""
    over = dict(over)
    fil = over.pop("_filament", None)
    full = over.pop("_full_bed", False)
    fpath = write_profiles(work / plate, fil)["filament"] if fil else base["filament"]
    mpath = full_bed_machine(base["machine"], work / plate / "machine_full_bed.json") if full else base["machine"]
    proc = json.loads(base["process"].read_text(encoding="utf-8"))
    proc.update(over)
    proc["curr_bed_type"] = BED
    proc["name"] = f"OpenVinyl turntable {plate}"
    pp = work / f"process_{plate}.json"
    pp.write_text(json.dumps(proc, indent=1), encoding="utf-8")
    out = dest / (out_name or f"plate_{plate}.3mf")
    models = centred_copies(stl, parts, work / plate / "stl") if full else [stl / f"{p}.stl" for p in parts]
    cmd = [str(STUDIO / "bambu-studio.exe"), "--orient", "0", "--arrange", "0" if full else "1", "--allow-rotations",
           "--load-settings", f"{mpath};{pp}", "--load-filaments", str(fpath),
           "--slice", "0", "--export-3mf", out.name, "--outputdir", str(dest)] + [str(m) for m in models]
    r = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if r.returncode != 0 or not out.exists():
        errs = [l for l in (r.stdout + r.stderr).splitlines() if "error" in l.lower()]
        raise RuntimeError(f"{plate}: FAILED (exit {r.returncode})" + chr(10) + chr(10).join(errs)[-3000:])
    (dest / "plate_1.gcode").unlink(missing_ok=True)   # the CLI's loose copy; the G-code is inside the 3MF
    s, g = summary(out)
    # guard: a broken mesh slices to almost nothing (Nov 22: a bad boolean left the platter STL as a 0.1 cm3
    # sliver and plate 2 came out at 0.2 g). Even at low infill a plate weighs a good part of its solid volume.
    import trimesh
    vol = sum(trimesh.load(str(stl / f"{q}.stl")).volume for q in parts) / 1000.0     # cm3
    if g < 0.3 * vol * 1.24:
        raise RuntimeError(f"{plate}: FAILED sanity: sliced {g:.1f} g for {vol:.1f} cm3 of parts; check the STLs")
    return s or 0, g


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    spine = mode in ("spine", "briefcase")
    stl, dest, plates = {"spine": (HERE.parent / "cad" / "stl_spine", HERE / "spine", SPINE_PLATES),
                         "briefcase": (HERE.parent / "cad" / "stl_briefcase", HERE / "briefcase", BRIEFCASE_PLATES)
                         }.get(mode, (STL, HERE, PLATES))
    dest.mkdir(exist_ok=True)
    work = HERE / "profiles"
    work.mkdir(exist_ok=True)
    base = write_profiles(work)
    total_s, total_g = 0, 0.0
    only = [a for a in sys.argv[2:]] if spine else []
    for plate, (parts, over, why) in plates.items():
        if only and plate.split("_")[0] not in only:
            continue
        try:
            s, g = slice_one(plate, parts, over, stl, dest, work, base)
        except RuntimeError as e:
            print(e)
            sys.exit(1)
        total_s += s
        total_g += g
        print(f"{plate:12s} {', '.join(parts):28s} {s / 3600:5.2f} h  {g:6.1f} g   {why}")
    print(f"{'total':12s} {'':28s} {total_s / 3600:5.2f} h  {total_g:6.1f} g")


if __name__ == "__main__":
    main()
