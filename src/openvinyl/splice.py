"""Splice the groove block into a slicer-exported base disc G-code.

Targets Bambu Studio output for the P1S but only relies on generic
structure: the last extrusion move marks the end of the base part, and the
end G-code starts at the first end-marker after it.
"""

import hashlib
import math
import re
import zipfile

_PARAM = re.compile(r"([XYZEF])(-?(?:\d+\.?\d*|\.\d+))")   # Bambu writes E.0123 (no leading zero)
_MOVE = re.compile(r"^G[01]\s")
_END_MARKERS = [
    re.compile(r"^; filament end gcode", re.I),
    re.compile(r"^;=+.*end", re.I),
    re.compile(r"^; EXECUTABLE_BLOCK_END", re.I),
    re.compile(r"^M140 S0\b"),
    re.compile(r"^M104 S0\b"),
    re.compile(r"^M400\s*;\s*wait", re.I),
]
_CFG = re.compile(r"^;\s*([a-z_]+)\s*=\s*(.+)$")


def _params(line: str) -> dict:
    code = line.split(";", 1)[0]
    return {k: float(v) for k, v in _PARAM.findall(code)}


def analyze_base(lines: list) -> dict:
    """Locate the insertion point and read what the block needs from the base."""
    info = {"extrusion_mode": "M83", "top_z": None, "last_extrusion_idx": None,
            "insert_idx": None, "center": None, "config": {}, "last_e_abs": 0.0,
            "layer_marker": None, "layer_count": 0}
    z = 0.0
    e_abs = 0.0
    mode = "M83"
    last_layer_start = 0
    xy_layer = []
    x = y = None
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("M82"):
            mode = "M82"
            continue
        if line.startswith("M83"):
            mode = "M83"
            continue
        if line.startswith("G92"):
            pr = _params(line)
            if "E" in pr:
                e_abs = pr["E"]
            continue
        if (line.startswith(";LAYER_CHANGE") or line.startswith("; CHANGE_LAYER")      # Prusa/Orca, Bambu Studio
                or line.startswith(";LAYER:")):                                       # Cura
            info["layer_marker"] = ("bambu" if line.startswith("; CHANGE_LAYER")
                                    else "cura" if line.startswith(";LAYER:") else "prusa")
            info["layer_count"] += 1
            last_layer_start = i
            xy_layer = []
            continue
        m = _CFG.match(line)
        if m and m.group(1) in ("nozzle_diameter", "layer_height", "printer_model",
                                "filament_type", "curr_bed_type"):
            info["config"][m.group(1)] = m.group(2).strip()
            continue
        if not _MOVE.match(line):
            continue
        pr = _params(line)
        if "Z" in pr:
            z = pr["Z"]
        if "X" in pr:
            x = pr["X"]
        if "Y" in pr:
            y = pr["Y"]
        if "E" in pr and ("X" in pr or "Y" in pr):
            if mode == "M83":
                extruding = pr["E"] > 0
            else:
                extruding = pr["E"] > e_abs
                e_abs = pr["E"]
            if extruding and x is not None and y is not None:
                info["last_extrusion_idx"] = i
                info["top_z"] = z
                xy_layer.append((x, y))
        elif "E" in pr and mode == "M82":
            e_abs = pr["E"]
    if info["last_extrusion_idx"] is None:
        raise ValueError("no extrusion moves found in base G-code")
    info["extrusion_mode"] = mode
    info["last_e_abs"] = e_abs
    if xy_layer:
        xs = [p[0] for p in xy_layer]
        ys = [p[1] for p in xy_layer]
        info["center"] = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
        info["last_layer_extent"] = (max(xs) - min(xs), max(ys) - min(ys))
    start = info["last_extrusion_idx"] + 1
    insert = None
    for j in range(start, len(lines)):
        s = lines[j].strip()
        if any(m.match(s) for m in _END_MARKERS):
            insert = j
            break
    info["insert_idx"] = start if insert is None else insert
    info["end_marker"] = None if insert is None else lines[insert].strip()
    return info


def _layer_header(info: dict, block_lines: list) -> list:
    """Mark the groove as one more layer in the base's own style, so slicer previews (which sort moves by these
    comments) draw it, and the printer's layer counter shows it. Nothing here moves the nozzle."""
    z = cur = None                                   # the height of the first extruding move (not the travel lift)
    for l in block_lines:
        if not _MOVE.match(l.strip()):
            continue
        pr = _params(l)
        cur = pr.get("Z", cur)
        if pr.get("E", 0) > 0 and ("X" in pr or "Y" in pr):
            z = cur
            break
    if info.get("layer_marker") is None or z is None:
        return []
    n = info["layer_count"] + 1
    h = info["config"].get("layer_height", "0.2").split(",")[0]
    if info["layer_marker"] == "bambu":
        return ["; CHANGE_LAYER", f"; Z_HEIGHT: {z:g}", f"; LAYER_HEIGHT: {h}",
                f"; layer num/total_layer_count: {n}/{n}", "; update layer progress", f"M73 L{n}",
                f"M991 S0 P{n - 1} ;notify layer change", "; FEATURE: Outer wall"]
    if info["layer_marker"] == "cura":                 # Cura counts layers from 0
        return [f";LAYER:{n - 1}", ";TYPE:WALL-OUTER"]
    return [";LAYER_CHANGE", f";Z:{z:g}", f";HEIGHT:{h}", ";TYPE:External perimeter"]


_DUR = re.compile(r"(?:(\d+)d\s*)?(?:(\d+)h\s*)?(?:(\d+)m\s*)?(?:(\d+)s)?")
_TIME_LINE = re.compile(r"^(; model printing time: )(.+?)(; total estimated time: )(.+)$")       # Bambu Studio
_TIME_LINE_PRUSA = re.compile(r"^(; estimated printing time \((?:normal|silent) mode\) = )(.+)$")  # Prusa/Orca
_M73 = re.compile(r"^M73((?:\s+[PRQS]-?\d+)+)\s*(;.*)?$")


def _seconds(text: str) -> float:
    m = _DUR.fullmatch(text.strip())
    if not m or not any(m.groups()):
        return None
    d, h, mi, se = (int(g) if g else 0 for g in m.groups())
    return float(((d * 24 + h) * 60 + mi) * 60 + se)


def _duration(sec: float) -> str:
    sec = int(round(sec))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return (f"{h}h {m}m {s}s" if h else f"{m}m {s}s")


def block_stats(block_lines: list) -> dict:
    """Time and filament of the groove pass from its own moves: each move's length over its (modal) feedrate,
    E-only moves by their E. No acceleration model: at 24-30 mm/s on 0.1 mm segments the planner holds speed."""
    x = y = z = None
    f = None
    t = 0.0
    e_mm = 0.0
    for raw in block_lines:
        line = raw.strip()
        if not _MOVE.match(line):
            continue
        pr = _params(line)
        f = pr.get("F", f)
        nx, ny, nz = pr.get("X", x), pr.get("Y", y), pr.get("Z", z)
        d = 0.0
        if x is not None and nx is not None and ny is not None:
            d = math.sqrt((nx - x) ** 2 + (ny - (y or 0.0)) ** 2 + ((nz or 0.0) - (z or 0.0)) ** 2)
        elif "E" in pr and not ("X" in pr or "Y" in pr):
            d = abs(pr["E"])
        if f:
            t += d / (f / 60.0)
        if pr.get("E", 0) > 0:
            e_mm += pr["E"]
        x, y, z = nx, ny, nz
    return {"time_s": t, "filament_mm": e_mm}


def _base_time_s(base_lines: list) -> float:
    for l in base_lines[:400]:
        m = _TIME_LINE.match(l.strip())
        if m:
            return _seconds(m.group(4))
        m = _TIME_LINE_PRUSA.match(l.strip())
        if m:
            return _seconds(m.group(2))
    for l in base_lines:
        m = _M73.match(l.strip())
        if m:
            kv = dict((t[0], int(t[1:])) for t in m.group(1).split())
            if "R" in kv:
                return kv["R"] * 60.0
    return None


def _retime(base_lines: list, idx: int, block_lines: list, stats: dict, info: dict) -> tuple:
    """Make the printer's clock include the groove: the header's time and filament totals, every progress line
    (M73 P percent R minutes left, and the silent-mode Q/S) in the base scaled to the new total, and progress
    lines through the groove. Returns (base_lines, block_lines)."""
    tb = _base_time_s(base_lines)
    tg = stats["time_s"]
    info["base_time_s"], info["groove_time_s"] = tb, tg
    if tb is None or tg <= 0:
        return base_lines, block_lines
    total = tb + tg
    dens = next((float(l.split(":", 1)[1]) for l in base_lines[:400] if l.startswith("; filament_density:")), 1.24)
    dia = next((float(l.split(":", 1)[1]) for l in base_lines[:400] if l.startswith("; filament_diameter:")), 1.75)
    vol = stats["filament_mm"] * math.pi * dia * dia / 4.0                     # mm3
    info["groove_filament_g"] = vol * dens / 1000.0
    info["total_time_s"] = total

    def add_header(l):
        m = _TIME_LINE.match(l)
        if m:
            mt, tt = _seconds(m.group(2)), _seconds(m.group(4))
            return f"{m.group(1)}{_duration((mt or 0) + tg)}{m.group(3)}{_duration((tt or 0) + tg)}"
        m = _TIME_LINE_PRUSA.match(l)
        if m:
            return f"{m.group(1)}{_duration((_seconds(m.group(2)) or 0) + tg)}"
        for key, add in (("; total filament length [mm] :", stats["filament_mm"]),
                         ("; total filament volume [cm^3] :", vol),            # Bambu writes mm3 here
                         ("; total filament weight [g] :", info["groove_filament_g"])):
            if l.startswith(key):
                try:
                    return f"{key} {float(l[len(key):]) + add:.2f}"
                except ValueError:
                    return l
        return l

    def scale_m73(l):
        m = _M73.match(l)
        if not m:
            return l
        kv = [(t[0], int(t[1:])) for t in m.group(1).split()]
        out = []
        for k, v in kv:
            if k in "PQ":
                v = int(round(v * tb / total))
            elif k in "RS":
                v = int(round(v + tg / 60.0))
            out.append(f"{k}{v}")
        return "M73 " + " ".join(out) + (f" {m.group(2)}" if m.group(2) else "")
    head = [add_header(l.strip()) if l.startswith("; ") else l for l in base_lines[:idx]]
    head = [scale_m73(l.strip()) if l.startswith("M73 ") else l for l in head]
    silent = any(l.startswith("M73 ") and " Q" in l for l in head)
    # progress through the groove: a line each time the percentage moves on
    out_block, t, last = [], 0.0, None
    st = {"x": None, "y": None, "z": None, "f": None}
    for raw in block_lines:
        line = raw.strip()
        if _MOVE.match(line):
            pr = _params(line)
            st["f"] = pr.get("F", st["f"])
            nx, ny, nz = pr.get("X", st["x"]), pr.get("Y", st["y"]), pr.get("Z", st["z"])
            d = 0.0
            if st["x"] is not None and nx is not None and ny is not None:
                d = math.sqrt((nx - st["x"]) ** 2 + (ny - (st["y"] or 0)) ** 2 + ((nz or 0) - (st["z"] or 0)) ** 2)
            elif "E" in pr and not ("X" in pr or "Y" in pr):
                d = abs(pr["E"])
            if st["f"]:
                t += d / (st["f"] / 60.0)
            st["x"], st["y"], st["z"] = nx, ny, nz
            pct = min(99, int((tb + t) / total * 100))
            if pct != last:
                rem = int(round((tg - t) / 60.0))
                out_block.append(f"M73 P{pct} R{rem}" + (f"\nM73 Q{pct} S{rem}" if silent else ""))
                last = pct
        out_block.append(raw)
    out_block = [x for l in out_block for x in l.split("\n")]
    return head + base_lines[idx:], out_block


def pack_3mf(base_3mf: str, gcode_text: str, out_path: str, info: dict) -> None:
    """The base disc's sliced Bambu project (.3mf from Bambu Studio's CLI or 'Export plate sliced file') with its
    plate G-code replaced by the spliced one: new md5, and the time, weight and filament totals the printer shows
    before it starts. The printer runs this G-code as it is; Bambu Studio would re-slice it and drop the groove."""
    data = gcode_text.encode()
    add_s = info.get("groove_time_s") or 0.0
    add_g = info.get("groove_filament_g") or 0.0
    add_m = (info.get("groove_filament_mm") or 0.0) / 1000.0
    with zipfile.ZipFile(base_3mf) as zin, zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zout:
        names = zin.namelist()
        if "Metadata/plate_1.gcode" not in names:
            raise ValueError(f"{base_3mf} holds no sliced plate (Metadata/plate_1.gcode)")
        for n in names:
            if n == "Metadata/plate_1.gcode":
                zout.writestr(n, data)
            elif n == "Metadata/plate_1.gcode.md5":
                zout.writestr(n, hashlib.md5(data).hexdigest().upper())
            elif n == "Metadata/slice_info.config":
                x = zin.read(n).decode()
                x = re.sub(r'(key="prediction" value=")(\d+)(")', lambda m: f"{m.group(1)}{int(m.group(2)) + int(round(add_s))}{m.group(3)}", x)
                x = re.sub(r'(key="weight" value=")([\d.]+)(")', lambda m: f"{m.group(1)}{float(m.group(2)) + add_g:.2f}{m.group(3)}", x)
                x = re.sub(r'(<filament id="1"[^>]*used_m=")([\d.]+)(")', lambda m: f"{m.group(1)}{float(m.group(2)) + add_m:.2f}{m.group(3)}", x, count=1)
                x = re.sub(r'(<filament id="1"[^>]*used_g=")([\d.]+)(")', lambda m: f"{m.group(1)}{float(m.group(2)) + add_g:.2f}{m.group(3)}", x, count=1)
                zout.writestr(n, x)
            else:
                zout.writestr(n, zin.read(n))


def splice(base_lines: list, block_lines: list, info: dict = None) -> tuple:
    info = analyze_base(base_lines) if info is None else info
    idx = info["insert_idx"]
    hdr = _layer_header(info, block_lines)
    if hdr and info["layer_marker"] == "bambu":     # the header's layer total, which previews size their slider by
        n = info["layer_count"]
        base_lines = [f"; total layer number: {n + 1}" if l.strip() == f"; total layer number: {n}" else l
                      for l in base_lines]
    if hdr and info["layer_marker"] == "cura":
        n = info["layer_count"]
        base_lines = [f";LAYER_COUNT:{n + 1}" if l.strip() == f";LAYER_COUNT:{n}" else l for l in base_lines]
    stats = block_stats(block_lines)
    info["groove_filament_mm"] = stats["filament_mm"]
    base_lines, block_lines = _retime(base_lines, idx, block_lines, stats, info)
    pre = ["", "; ---- OpenVinyl groove spliced here ----", "M400 ; finish base disc"] + hdr
    post = []
    if info["extrusion_mode"] == "M82":
        post += ["M82 ; restore absolute E", f"G92 E{info['last_e_abs']:.5f}"]
    post += ["; ---- end OpenVinyl groove ----", ""]
    return base_lines[:idx] + pre + block_lines + post + base_lines[idx:], info
