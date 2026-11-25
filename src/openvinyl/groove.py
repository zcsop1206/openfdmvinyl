"""Lateral-cut groove model.

The groove follows an Archimedean spiral r(theta) = r_outer - (p / 2pi) theta.
Audio modulates the groove radially: x(theta) = A * audio(t), t = theta / omega.
Two beads per turn, centre spacing s, form a V-crease the stylus rides in.

Everything here is geometry and physics. No G-code.
"""

from dataclasses import dataclass, field, asdict
import math
import numpy as np

from .printer import PrinterProfile


@dataclass
class GrooveParams:
    rpm: float = 78.0
    record_diameter: float = 254.0
    outer_groove_radius: float = 120.0
    inner_groove_radius: float = 40.0
    amplitude: float = 0.10            # A, mm, peak radial modulation
    bead_spacing_factor: float = 1.1   # s = factor * w_e
    land: float = None                 # gap between turns, default w_e
    lead_in_turns: float = 1.0         # unmodulated turns at the start
    lead_out_turns: float = 1.0        # unmodulated turns at the end
    segment_length: float = 0.10       # ds, mm of arc per G-code segment
    theta_max_deg: float = 45.0        # max trackable groove slope (open parameter)
    groove_layers: int = 1             # bead layers stacked per bead
    extra_layers_from_s: float = None  # layers above the first start at this play time (None: whole groove)
    floor_layers: int = 0              # spiral floor layers under the groove, following it (covers the
                                       # base's straight top lines, which play as a ridge texture: Nov 22)
    fade_s: float = 0.2                # fade in/out of modulation, seconds
    schedule: list = None              # bands along the groove, in order: dicts with t1 (audio time, s, where the
                                       # band ends; the last band runs to the end), spacing (bead spacing factor),
                                       # layers and floor (layer counts). Overrides bead_spacing_factor,
                                       # groove_layers, extra_layers_from_s and floor_layers. (Taguchi disc, Nov 24)
    ramp_s: float = 0.75               # spacing ramps, and the floor steps a layer at a time, over this long
                                       # centred on each band end
    stylus_radius: float = 0.076       # mm, 3 mil tip, used for reporting and preview

    def resolve(self, printer: PrinterProfile) -> "GrooveGeometry":
        return GrooveGeometry(self, printer)


class GrooveGeometry:
    """Derived quantities. Pure functions of params + printer."""

    def __init__(self, p: GrooveParams, printer: PrinterProfile):
        self.p = p
        self.printer = printer
        self.w_e = printer.extrusion_width
        # with a schedule the pitch is set by the widest band, so every band keeps the same spiral
        f_max = max(b["spacing"] for b in p.schedule) if p.schedule else p.bead_spacing_factor
        self.bead_spacing = f_max * self.w_e
        self.land = self.w_e if p.land is None else p.land
        self.pitch = self.bead_spacing + self.w_e + 2.0 * p.amplitude + self.land
        self.k = self.pitch / (2.0 * math.pi)          # dr/dtheta magnitude
        self.omega = 2.0 * math.pi * p.rpm / 60.0       # rad/s
        self.radial_travel = p.outer_groove_radius - p.inner_groove_radius
        self.turns_available = self.radial_travel / self.pitch
        self.theta_end = 2.0 * math.pi * self.turns_available
        self.time_available = self.theta_end / self.omega
        self.lead_in_time = 2.0 * math.pi * p.lead_in_turns / self.omega
        self.lead_out_time = 2.0 * math.pi * p.lead_out_turns / self.omega
        self.audio_time_available = max(
            0.0, self.time_available - self.lead_in_time - self.lead_out_time)

    # -- spiral --------------------------------------------------------------
    def r_of_theta(self, theta):
        return self.p.outer_groove_radius - self.k * theta

    def theta_of_arc(self, s):
        """Invert arc length s(theta) = r_out*theta - (k/2)*theta^2
        (the dr/dtheta contribution to arc length is negligible: k << r)."""
        r0 = self.p.outer_groove_radius
        disc = r0 * r0 - 2.0 * self.k * s
        disc = np.maximum(disc, 0.0)
        return (r0 - np.sqrt(disc)) / self.k

    def arc_of_theta(self, theta):
        return self.p.outer_groove_radius * theta - 0.5 * self.k * theta * theta

    def v_play(self, r):
        """Groove linear speed under the stylus, mm/s."""
        return self.omega * r

    # -- physics limits ------------------------------------------------------
    def f_smear(self, r):
        """Nozzle smearing along the path: features shorter than one bead
        width are averaged out. First-order bandwidth limit."""
        return self.v_play(r) / self.w_e

    def f_stylus(self, r):
        """Stylus geometric filter, f = v / (2 pi R)."""
        return self.v_play(r) / (2.0 * math.pi * self.p.stylus_radius)

    def af_slope_limit(self, r):
        """A * f product allowed by the slope constraint, mm * Hz."""
        return math.tan(math.radians(self.p.theta_max_deg)) * self.v_play(r) / (2.0 * math.pi)

    def summary(self) -> dict:
        p = self.p
        ri, ro = p.inner_groove_radius, p.outer_groove_radius
        return {
            "rpm": p.rpm,
            "extrusion_width_mm": round(self.w_e, 4),
            "bead_spacing_mm": round(self.bead_spacing, 4),
            "amplitude_mm": p.amplitude,
            "land_mm": round(self.land, 4),
            "pitch_mm": round(self.pitch, 4),
            "turns_available": round(self.turns_available, 2),
            "time_available_s": round(self.time_available, 2),
            "audio_time_available_s": round(self.audio_time_available, 2),
            "v_play_inner_mm_s": round(self.v_play(ri), 1),
            "v_play_outer_mm_s": round(self.v_play(ro), 1),
            "f_smear_inner_hz": round(self.f_smear(ri), 1),
            "f_smear_outer_hz": round(self.f_smear(ro), 1),
            "f_stylus_inner_hz": round(self.f_stylus(ri), 1),
            "f_stylus_outer_hz": round(self.f_stylus(ro), 1),
            "af_slope_limit_inner_mm_hz": round(self.af_slope_limit(ri), 1),
            "f_full_amplitude_inner_hz": round(self.af_slope_limit(ri) / p.amplitude, 1),
            "segment_length_mm": p.segment_length,
            "segments_per_rev_inner": int(2 * math.pi * ri / p.segment_length),
            "segments_per_rev_outer": int(2 * math.pi * ro / p.segment_length),
        }


@dataclass
class GroovePath:
    """Sampled groove: one entry per G-code vertex along the spiral."""
    theta: np.ndarray        # rad, unwrapped, increasing
    r_spiral: np.ndarray     # unmodulated spiral radius
    x: np.ndarray            # radial modulation, mm
    t: np.ndarray            # play time at each vertex, s
    speed: np.ndarray        # commanded feedrate per vertex (into that vertex), mm/s
    geometry: GrooveGeometry
    stats: dict = field(default_factory=dict)
    spacing: np.ndarray = None   # bead centre spacing per vertex, mm (None: geometry.bead_spacing everywhere)
    layers: np.ndarray = None    # bead layers per vertex
    floor: np.ndarray = None     # floor layers under each vertex
    floor_z: np.ndarray = None   # floor height under each vertex in layers (the beads sit on it)

    @property
    def r_center(self):
        return self.r_spiral + self.x

    def bead_xy(self, cx=0.0, cy=0.0):
        """Bead centrelines (L outer, R inner) as ((xL, yL), (xR, yR)).

        Each bead sits s/2 from the groove centreline along its in-plane
        normal. Offsetting radially instead would shrink the perpendicular
        gap to s*cos(slope) and close the crease on steep passages."""
        half = (self.spacing if self.spacing is not None else self.geometry.bead_spacing) / 2.0
        (xL, yL), (xR, yR) = offset_beads(self.theta, self.r_center, half)
        return (xL + cx, yL + cy), (xR + cx, yR + cy)

    def bead_paths(self, cx=0.0, cy=0.0):
        """Bead toolpaths ready to emit, [(x, y, v) for L, R]. Both beads keep
        the centreline's vertex count, so vertex i of L and R straddle
        centreline vertex i (the decoder relies on this). Feedrate scales with
        each bead's own segment length, capped at the centreline feedrate, so
        the inner bead never exceeds the segment-rate limit on tight bends."""
        out = []
        ds = self.geometry.p.segment_length
        for bx, by in self.bead_xy(cx, cy):
            seg = np.concatenate([[ds], np.hypot(np.diff(bx), np.diff(by))])
            out.append((bx, by, self.speed * np.minimum(1.0, seg / ds)))
        return out

    def xy(self, r, cx=0.0, cy=0.0):
        return cx + r * np.cos(self.theta), cy + r * np.sin(self.theta)


def offset_beads(theta: np.ndarray, r_c: np.ndarray, half: float):
    """Offset the polar curve r_c(theta) by +-half along its normal.
    The tangent is the central-difference chord of the sampled polyline
    (its vertex bisector at constant segment length), so the offsets are
    parallel to the segments actually printed.
    Returns ((x_out, y_out), (x_in, y_in)) about the origin."""
    px, py = r_c * np.cos(theta), r_c * np.sin(theta)
    tx, ty = np.gradient(px), np.gradient(py)
    tn = np.hypot(tx, ty)
    nx, ny = ty / tn, -tx / tn          # tangent rotated -90 deg: outward for a CCW spiral
    return (px + half * nx, py + half * ny), (px - half * nx, py - half * ny)


def crease_checks(path: "GroovePath") -> dict:
    """Measured gap between the two bead centrelines and whether the inner
    offset folds back on itself (cusp where curvature radius < s/2).
    crease_gap_short_mm: the most the gap falls short of the spacing asked for at that vertex."""
    from scipy.spatial import cKDTree
    (xL, yL), (xR, yR) = path.bead_xy()
    gap, _ = cKDTree(np.column_stack([xR, yR])).query(np.column_stack([xL, yL]))
    want = path.spacing if path.spacing is not None else np.full(len(gap), path.geometry.bead_spacing)
    # inner bead must advance in the same direction as the centreline everywhere
    xc, yc = path.r_center * np.cos(path.theta), path.r_center * np.sin(path.theta)
    dot = np.diff(xR) * np.diff(xc) + np.diff(yR) * np.diff(yc)
    return {"crease_gap_min_mm": float(gap.min()), "crease_gap_short_mm": float(np.max(want - gap)),
            "inner_bead_reversals": int(np.sum(dot <= 0))}


def schedule_fields(params: GrooveParams, w_e: float, t_audio: np.ndarray) -> dict:
    """Per-vertex bead spacing (mm), bead layers, floor layers and ramped floor height (layers) at audio time
    t_audio. Without a schedule: the fixed parameters (with extra_layers_from_s, the layers above the first start
    there). With one: each band's values; spacing and floor height ramp linearly over ramp_s centred on each band
    end, and layer counts switch at the band end itself."""
    n = len(t_audio)
    if not params.schedule:
        layers = np.full(n, params.groove_layers)
        if params.extra_layers_from_s is not None:
            layers = np.where(t_audio >= params.extra_layers_from_s, params.groove_layers, 1)
        fl = np.full(n, params.floor_layers)
        return {"spacing": np.full(n, params.bead_spacing_factor * w_e), "layers": layers, "floor": fl,
                "floor_z": fl.astype(float)}
    bands = params.schedule
    ends = np.array([b.get("t1", np.inf) for b in bands[:-1]], dtype=float)
    k = np.searchsorted(ends, t_audio, side="right")                     # band index per vertex
    val = lambda key: np.array([b[key] for b in bands], dtype=float)
    out = {"layers": val("layers")[k].astype(int), "floor": val("floor")[k].astype(int)}
    half = params.ramp_s / 2.0
    for key in ("spacing", "floor"):
        v = val(key)
        y = v[k].copy()
        for i, te in enumerate(ends):                                    # ramp across each band end
            m = np.abs(t_audio - te) < half
            if half > 0 and m.any():
                a = (t_audio[m] - (te - half)) / (2 * half)
                y[m] = v[i] + (v[i + 1] - v[i]) * a
        out[key] = y
    # the floor changes one whole layer at a time (a staircase across the ramp), and the beads sit on that same
    # count: a smooth bead ramp over a stepped floor would print in air on one side and into the floor on the other
    out["floor"] = np.round(out["floor"]).astype(int)
    out["floor_z"] = out["floor"].astype(float)
    out["spacing"] = out["spacing"] * w_e
    return out


def _running_min(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return a
    from scipy.ndimage import minimum_filter1d
    return minimum_filter1d(a, size=w, mode="nearest")


def _moving_average(a: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return a
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(a, size=w, mode="nearest")


def _d2(y, ds):
    return np.gradient(np.gradient(y, ds), ds)


def curvature_limit(x: np.ndarray, ds: float, kappa_max: float, window: int, passes: int = 30):
    """Same gain method as slope_limit, on |d2x/ds2| (groove bend relative to
    the spiral). Keeps the inner bead's nozzle path from folding back where
    the groove's bend radius drops below half the bead spacing."""
    y = x.copy()
    reduced = np.zeros(len(x), dtype=bool)
    target = kappa_max * 1.01
    for _ in range(passes):
        k = np.abs(_d2(y, ds))
        if k.max() <= target:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(k > kappa_max, kappa_max / k, 1.0)
        reduced |= g < 1.0
        g = _running_min(g, window)
        g = _moving_average(_moving_average(g, window), window)
        y = y * g
    k_max = np.abs(_d2(y, ds)).max()
    if k_max > target:
        y *= kappa_max / k_max
        reduced[:] = True
    return y, float(reduced.mean())


def slope_limit(x: np.ndarray, ds: float, tan_max: float, window: int, passes: int = 30):
    """Reduce local amplitude wherever |dx/ds| exceeds tan_max.

    Each pass: per-sample allowed gain, running minimum over `window`,
    smoothed twice (triangular kernel) so the gain envelope itself adds
    little slope, then applied. Iterates until the slope target is met.
    A final global scale guarantees the bound in the worst case.
    Returns (x_limited, fraction_of_samples_reduced)."""
    y = x.copy()
    reduced = np.zeros(len(x), dtype=bool)
    target = tan_max * 1.01
    for _ in range(passes):
        slope = np.abs(np.gradient(y, ds))
        if slope.max() <= target:
            break
        with np.errstate(divide="ignore", invalid="ignore"):
            g = np.where(slope > tan_max, tan_max / slope, 1.0)
        reduced |= g < 1.0
        g = _running_min(g, window)
        g = _moving_average(_moving_average(g, window), window)
        y = y * g
    slope_max = np.abs(np.gradient(y, ds)).max()
    if slope_max > target:
        y *= tan_max / slope_max
        reduced[:] = True
    return y, float(reduced.mean())


def menger_curvature(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Curvature at each interior vertex from its neighbours. Ends copy."""
    ax, ay = x[:-2], y[:-2]
    bx, by = x[1:-1], y[1:-1]
    cx, cy = x[2:], y[2:]
    ab = np.hypot(bx - ax, by - ay)
    bc = np.hypot(cx - bx, cy - by)
    ca = np.hypot(cx - ax, cy - ay)
    area2 = np.abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
    with np.errstate(divide="ignore", invalid="ignore"):
        k = np.where(ab * bc * ca > 0, 2.0 * area2 / (ab * bc * ca), 0.0)
    return np.concatenate([[k[0]], k, [k[-1]]])


def build_groove(audio: np.ndarray, fs: float, params: GrooveParams,
                 printer: PrinterProfile) -> GroovePath:
    """Sample the spiral at constant arc length, map audio onto it, apply the
    slope limiter, and schedule feedrate from path curvature."""
    g = params.resolve(printer)
    ds = params.segment_length
    stats = {}

    audio_dur = len(audio) / fs
    if audio_dur > g.audio_time_available:
        stats["truncated_audio_s"] = round(audio_dur - g.audio_time_available, 2)
        audio = audio[: int(g.audio_time_available * fs)]
        audio_dur = len(audio) / fs
    total_time = g.lead_in_time + audio_dur + g.lead_out_time
    theta_end = total_time * g.omega
    s_end = g.arc_of_theta(theta_end)

    n = int(s_end / ds) + 1
    s = np.arange(n) * ds
    theta = g.theta_of_arc(s)
    r = g.r_of_theta(theta)
    t = theta / g.omega

    # Modulation: audio interpolated at play time, with fades.
    t_audio = t - g.lead_in_time
    x = np.zeros(n)
    if len(audio) > 0:
        idx = t_audio * fs
        valid = (idx >= 0) & (idx < len(audio) - 1)
        x[valid] = np.interp(idx[valid], np.arange(len(audio)), audio)
        fade = params.fade_s
        if fade > 0:
            env = np.clip(t_audio / fade, 0, 1) * np.clip((audio_dur - t_audio) / fade, 0, 1)
            x *= np.clip(env, 0, 1)
    x *= params.amplitude

    # Slope limiter (stylus tracking constraint).
    tan_max = math.tan(math.radians(params.theta_max_deg))
    win = max(3, int(round(1.0 / ds)))          # 1 mm of arc
    x, frac = slope_limit(x, ds, tan_max, win)
    stats["slope_limited_fraction"] = round(frac, 4)
    # Curvature limiter (two-bead printability): the groove's bend radius,
    # including the spiral's own 1/r, stays at or above the bead spacing s.
    # At s/2 the inner bead's nozzle path would fold back on itself; at s its
    # segments shrink to half length at most, keeping them >= 0.05 mm.
    kappa_max = 0.7 / g.bead_spacing - 1.0 / params.inner_groove_radius   # 30 %: d2x underestimates at 3-5 samples per wavelength
    x, frac_k = curvature_limit(x, ds, kappa_max, win)
    stats["curvature_limited_fraction"] = round(frac_k, 4)
    stats["min_bend_radius_mm"] = round(1.0 / (np.abs(_d2(x, ds)).max() + 1.0 / params.inner_groove_radius), 4)
    stats["max_slope_deg"] = round(math.degrees(math.atan(np.abs(np.gradient(x, ds)).max())), 2)

    sf = schedule_fields(params, g.w_e, t - g.lead_in_time)

    # Feedrate from curvature of the beads; the inner one bends hardest.
    (xo, yo), (xi, yi) = offset_beads(theta, r + x, sf["spacing"] / 2.0)
    kappa = np.maximum(menger_curvature(xo, yo), menger_curvature(xi, yi))
    with np.errstate(divide="ignore"):
        v_curv = np.where(kappa > 0, np.sqrt(printer.groove_accel_max / kappa), np.inf)
    v_seg = printer.max_segment_rate * ds
    v_flow = printer.max_volumetric_flow / printer.bead_area()
    v = np.minimum.reduce([v_curv, np.full(n, v_seg), np.full(n, v_flow),
                           np.full(n, printer.groove_speed_max)])
    v = np.maximum(v, printer.groove_speed_min)
    v = _running_min(v, max(3, int(round(1.0 / ds))))   # 1 mm lookahead
    v = np.round(v * 2.0) / 2.0                          # 0.5 mm/s steps

    stats.update({
        "vertices": n,
        "path_length_mm": round(float(s_end), 1),
        "play_time_s": round(total_time, 2),
        "audio_time_s": round(audio_dur, 2),
        "turns_used": round(theta_end / (2 * math.pi), 2),
        "r_end_mm": round(float(r[-1]), 2),
        "speed_min_mm_s": float(v.min()),
        "speed_max_mm_s": float(v.max()),
        "speed_mean_mm_s": round(float(v.mean()), 2),
        "limits_hit": {
            "curvature_vertices": int(np.sum(v_curv < printer.groove_speed_max)),
            "segment_rate_binding": bool(v_seg < min(v_flow, printer.groove_speed_max)),
            "flow_binding": bool(v_flow < min(v_seg, printer.groove_speed_max)),
        },
    })
    return GroovePath(theta=theta, r_spiral=r, x=x, t=t, speed=v, geometry=g, stats=stats,
                      spacing=sf["spacing"], layers=sf["layers"], floor=sf["floor"], floor_z=sf["floor_z"])
