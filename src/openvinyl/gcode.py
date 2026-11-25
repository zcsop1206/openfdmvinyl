"""G-code emission for the groove pass, with self-checks.

The emitted block is self-describing: every parameter needed to decode the
groove back into audio is written as `; OV key=value` comment lines, and
each bead pass is announced with `; OV bead=<L|R> layer=<n> dir=<in|out>`.
preview.py relies on exactly this format.
"""

from dataclasses import dataclass, field
import math
import numpy as np

from . import __version__
from .printer import PrinterProfile
from .groove import GroovePath, crease_checks

BLOCK_START = "; OPENVINYL_GROOVE_START"
BLOCK_END = "; OPENVINYL_GROOVE_END"

RETRACT_MM = 0.8
RETRACT_SPEED = 30.0     # mm/s
Z_SPEED = 10.0           # mm/s
PLA_DENSITY = 1.24       # g/cm^3
FLOOR_SPEED = 60.0       # mm/s, spiral floor
FLOOR_CHORD = 1.0        # mm of arc per floor segment (sagitta 0.003 mm at r 45)
FLOOR_EXT_TURNS = 1.0    # floor turns beyond the groove's first and last turn


@dataclass
class Report:
    checks: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def check(self, name, ok, value=None, limit=None, note=""):
        self.checks.append({"name": name, "ok": bool(ok), "value": value,
                            "limit": limit, "note": note})

    @property
    def ok(self):
        return all(c["ok"] for c in self.checks)

    def text(self) -> str:
        out = []
        for c in self.checks:
            mark = "PASS" if c["ok"] else "FAIL"
            v = "" if c["value"] is None else f" value={c['value']}"
            l = "" if c["limit"] is None else f" limit={c['limit']}"
            n = f"  ({c['note']})" if c["note"] else ""
            out.append(f"  [{mark}] {c['name']}{v}{l}{n}")
        out.append("  stats:")
        for k, v in self.stats.items():
            out.append(f"    {k}: {v}")
        return "\n".join(out)

    def to_dict(self):
        return {"ok": self.ok, "checks": self.checks, "stats": self.stats}


def _f(v, nd=3):
    s = f"{v:.{nd}f}"
    return s.rstrip("0").rstrip(".") if "." in s else s


def emit_groove(path: GroovePath, printer: PrinterProfile, z_top: float,
                center=(None, None)) -> tuple:
    """Return (lines, report). Coordinates are absolute, extrusion relative (M83)."""
    g = path.geometry
    p = g.p
    cx = printer.center[0] if center[0] is None else center[0]
    cy = printer.center[1] if center[1] is None else center[1]
    h = printer.layer_height
    e_mm = printer.e_per_mm()
    rep = Report()
    lines = []
    w = lines.append

    bead_L, bead_R = path.bead_paths(cx, cy)
    r_L = np.hypot(bead_L[0] - cx, bead_L[1] - cy)
    beads = [("L", bead_L, "in"), ("R", bead_R, "out")]

    # ---- header -----------------------------------------------------------
    w(BLOCK_START)
    hdr = {
        "version": __version__, "rpm": p.rpm, "center_x": cx, "center_y": cy,
        "outer_radius": p.outer_groove_radius, "inner_radius": p.inner_groove_radius,
        "pitch": g.pitch, "bead_spacing": g.bead_spacing, "amplitude": p.amplitude,
        "land": g.land, "extrusion_width": g.w_e, "layer_height": h,
        "segment_length": p.segment_length, "z_top": z_top, "layers": p.groove_layers,
        "stylus_radius": p.stylus_radius, "theta_max_deg": p.theta_max_deg,
        "lead_in_turns": p.lead_in_turns, "lead_out_turns": p.lead_out_turns,
        "floor_layers": p.floor_layers, "extra_layers_from_s": p.extra_layers_from_s,
        "schedule": p.schedule, "ramp_s": p.ramp_s if p.schedule else None,
        "play_time_s": path.stats.get("play_time_s"),
    }
    for k, v in hdr.items():
        w(f"; OV {k}={v}")
    w("G90 ; absolute XY")
    w("M83 ; relative E")

    total_len = 0.0
    total_e = 0.0
    print_time = 0.0
    min_seg = math.inf
    max_flow = 0.0
    max_seg_rate = 0.0
    out_of_bounds = 0
    n_segments = 0

    floor = path.floor if path.floor is not None else np.full(len(path.theta), p.floor_layers)
    floor_z = path.floor_z if path.floor_z is not None else floor.astype(float)
    layers = path.layers if path.layers is not None else np.full(len(path.theta), p.groove_layers)
    nf = int(floor.max())
    n_layers = int(layers.max())
    z_travel = z_top + (nf + n_layers) * h + printer.z_hop
    w(f"G1 Z{_f(z_travel)} F{int(Z_SPEED * 60)}")

    def runs(mask):
        """Index ranges [i0, i1) where mask holds."""
        d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
        return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))

    def extrude(xs, ys, zs, e_per_mm, f_of):
        """One extruding run: travel, lower, prime, the moves, retract, lift. f_of(i) = feedrate into vertex i
        (mm/s). zs: one height, or one per vertex (a Z word only where it changes). Returns the run's stats."""
        nonlocal total_len, total_e, print_time, min_seg, max_flow, max_seg_rate, out_of_bounds, n_segments
        zv = np.broadcast_to(np.asarray(zs, dtype=float), xs.shape)
        w(f"G1 X{_f(xs[0])} Y{_f(ys[0])} F{int(printer.travel_speed * 60)}")
        w(f"G1 Z{_f(zv[0])} F{int(Z_SPEED * 60)}")
        w(f"G1 E{_f(RETRACT_MM, 4)} F{int(RETRACT_SPEED * 60)} ; unretract")
        print_time += 2.0 * printer.z_hop / Z_SPEED + 2 * RETRACT_MM / RETRACT_SPEED
        seg = np.hypot(np.diff(xs), np.diff(ys))
        last_f, last_z = None, round(float(zv[0]), 3)
        for i in range(1, len(xs)):
            L = seg[i - 1]
            if L <= 0:
                continue
            if not printer.in_bounds(xs[i], ys[i]):
                out_of_bounds += 1
            vi = f_of(i)
            fi = int(round(vi * 60))
            zi = round(float(zv[i]), 3)
            zw = f" Z{_f(zi)}" if zi != last_z else ""
            e = e_per_mm * L
            fw = f" F{fi}" if fi != last_f else ""
            w(f"G1 X{_f(xs[i])} Y{_f(ys[i])}{zw} E{_f(e, 5)}{fw}")
            last_f, last_z = fi, zi
            total_len += L
            total_e += e
            print_time += L / vi
            n_segments += 1
            min_seg = min(min_seg, L)
            max_flow = max(max_flow, printer.volumetric_flow(vi) * e_per_mm / e_mm)
            max_seg_rate = max(max_seg_rate, vi / L)
        w(f"G1 E-{_f(RETRACT_MM, 4)} F{int(RETRACT_SPEED * 60)} ; retract")
        w(f"G1 Z{_f(z_travel)} F{int(Z_SPEED * 60)}")

    # ---- spiral floor: flat layers under the groove band, so its floor has no lines crossing it (the base's
    # straight top infill played as a 0.77 mm ridge texture, Nov 22). The lines follow the groove's own spiral:
    # n lines per turn, each a copy of the unmodulated groove spiral shifted inward by j * pitch / n, so line 0
    # lies centred under the crease all the way round and no seam between floor lines runs under the stylus
    # (the old floor, one spiral at one bead width per turn, drifted against the groove and put seams under some
    # turns' creases, Nov 24). Line width pitch / n, extrusion scaled to match, so the top is flat. Every layer
    # stacks the same way (the top one is the one the stylus sees, whatever the count under that band).
    if nf:
        P = g.pitch
        n_lines = max(1, int(round(P / g.w_e)))
        wf = P / n_lines
        ext = FLOOR_EXT_TURNS * 2 * math.pi
        th_a, th_b = float(path.theta[0]) - ext, float(path.theta[-1]) + ext
        r_min = g.r_of_theta(th_b) - P
        th = np.arange(th_a, th_b, FLOOR_CHORD / max(r_min, 1.0))
        rs = g.r_of_theta(th)
        e_floor = e_mm * wf / g.w_e
        rep.stats["floor"] = {"lines_per_turn": n_lines, "line_width_mm": round(wf, 4),
                              "band_mm": (round(float(rs.min() - P), 2), round(float(rs.max() + wf / 2), 2))}
        for k in range(nf):
            w(f"; OV floor layer={k}")
            z = z_top + (k + 1) * h
            for j in range(n_lines):
                # the floor count of the groove turn whose beads sit on this line at this angle: lines in the
                # inner half of the pitch carry the next turn inward's outer bead
                th_eff = np.clip(th + 2 * math.pi * round(j / n_lines), path.theta[0], path.theta[-1])
                cnt = floor[np.clip(np.searchsorted(path.theta, th_eff), 0, len(floor) - 1)]
                rr = rs - j * wf
                fx, fy = cx + rr * np.cos(th), cy + rr * np.sin(th)
                for a, b in runs(cnt > k):
                    if b - a < 3:
                        continue
                    xs, ys = (fx[a:b], fy[a:b]) if k % 2 == 0 else (fx[a:b][::-1], fy[a:b][::-1])
                    w(f"; OV floor line={j} layer={k}")
                    extrude(xs, ys, z, e_floor, lambda i: FLOOR_SPEED)

    # ---- beads: each layer printed where that many layers are wanted, sitting on the floor height under it
    for layer in range(n_layers):
        rr = runs(layers > layer)
        for name, (xs, ys, v), direction in beads:
            seq = rr[::-1] if direction == "out" else rr
            for a, b in seq:
                if b - a < 3:
                    continue
                zs = z_top + (floor_z[a:b] + layer + 1) * h
                bx, by, bv = xs[a:b], ys[a:b], v[a:b]
                if direction == "out":
                    bx, by, bv, zs = bx[::-1], by[::-1], bv[::-1], zs[::-1]
                w(f"; OV bead={name} layer={layer} dir={direction}")
                extrude(bx, by, zs, e_mm, lambda i, bv=bv: bv[i])
    w(BLOCK_END)

    # ---- checks -----------------------------------------------------------
    tan_max = math.tan(math.radians(p.theta_max_deg))
    slope = np.abs(np.gradient(path.x, p.segment_length)).max()
    rep.check("xy_in_bounds", out_of_bounds == 0, out_of_bounds, 0, "vertices outside bed")
    rep.check("z_in_range", z_travel <= printer.max_z, round(z_travel, 3), printer.max_z)
    rep.check("min_segment_length", min_seg >= printer.min_segment_length,
              round(min_seg, 4), printer.min_segment_length)
    rep.check("max_volumetric_flow", max_flow <= printer.max_volumetric_flow + 1e-9,
              round(max_flow, 3), printer.max_volumetric_flow, "mm^3/s")
    rep.check("max_segment_rate", max_seg_rate <= printer.max_segment_rate + 1e-6,
              round(max_seg_rate, 1), printer.max_segment_rate, "segments/s")
    rep.check("speed_range", printer.groove_speed_min - 1e-9 <= path.speed.min()
              and path.speed.max() <= printer.groove_speed_max + 1e-9,
              (float(path.speed.min()), float(path.speed.max())),
              (printer.groove_speed_min, printer.groove_speed_max))
    rep.check("land_width", g.land >= 0.5 * g.w_e, round(g.land, 3), round(0.5 * g.w_e, 3),
              "land below half a bead: turns may touch")
    cr = crease_checks(path)
    rep.check("crease_gap", cr["crease_gap_short_mm"] <= 0.005,
              round(cr["crease_gap_min_mm"], 4), round(float((path.spacing if path.spacing is not None
                                                                else [g.bead_spacing]).min()), 4),
              "min bead centre distance vs the smallest spacing asked for, mm; below w_e the crease closes")
    rep.check("inner_bead_no_cusp", cr["inner_bead_reversals"] == 0,
              cr["inner_bead_reversals"], 0, "inner offset folds where curvature radius < s/2")
    rep.check("groove_slope", slope <= tan_max * 1.02, round(math.degrees(math.atan(slope)), 2),
              p.theta_max_deg, "deg, after slope limiter")
    rep.check("inner_radius_respected", path.r_spiral[-1] >= p.inner_groove_radius - 1e-6,
              round(float(path.r_spiral[-1]), 2), p.inner_groove_radius)
    rep.check("record_fits_bed",
              p.record_diameter / 2.0 <= min(cx, cy, printer.bed_x - cx, printer.bed_y - cy) + 1e-6,
              p.record_diameter, round(2.0 * min(cx, cy, printer.bed_x - cx, printer.bed_y - cy), 1),
              "record diameter vs distance to nearest bed edge")
    rep.check("outer_bead_inside_record",
              r_L.max() + g.w_e / 2.0 <= p.record_diameter / 2.0,
              round(float(r_L.max() + g.w_e / 2.0), 2), p.record_diameter / 2.0)

    rep.stats.update({
        "groove_lines": len(lines),
        "segments": n_segments,
        "path_length_mm": round(total_len, 1),
        "filament_mm": round(total_e, 1),
        "filament_g": round(total_e * printer.filament_area / 1000.0 * PLA_DENSITY, 2),
        "groove_print_time_min": round(print_time / 60.0, 1),
        "f_smear_inner_hz": round(g.f_smear(p.inner_groove_radius), 1),
        "f_smear_outer_hz": round(g.f_smear(p.outer_groove_radius), 1),
        "f_stylus_inner_hz": round(g.f_stylus(p.inner_groove_radius), 1),
    })
    rep.stats.update(path.stats)
    return lines, rep


def generic_start(bed_temp=55, hotend_temp=220, printer: PrinterProfile = None):
    return [
        "; OPENVINYL standalone file. GENERIC start G-code, not the Bambu start sequence.",
        "; Prefer --base with a Bambu Studio export. This path is for testing on other firmware.",
        f"M140 S{bed_temp}", f"M104 S{hotend_temp}", "G28", f"M190 S{bed_temp}",
        f"M109 S{hotend_temp}", "G90", "M83", "G92 E0",
    ]


def generic_end():
    return ["M104 S0", "M140 S0", "G91", "G1 Z10 F600", "G90", "M84",
            "; OPENVINYL standalone end"]
