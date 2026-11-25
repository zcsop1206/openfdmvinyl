"""Decode a groove block back into audio and pictures, from the G-code alone.

This is simulation stage 1: a round trip through the emitted file. It
reads the `; OV` header, reconstructs both bead centrelines, takes the
crease between them as the stylus path, and applies a first-order playback
model (nozzle smear along the path, stylus geometric lowpass). Stage 2
(bead deposition heightmap + sphere stylus) replaces the playback model.
"""

import math
import re
import numpy as np

from .gcode import BLOCK_START, BLOCK_END
from .dsp import write_wav, lowpass

_PARAM = re.compile(r"([XYZEF])(-?\d+\.?\d*)")


def parse_block(lines: list) -> dict:
    """Return {'header': {...}, 'beads': [{'name','layer','dir','x','y','f'}]}."""
    try:
        i0 = next(i for i, l in enumerate(lines) if l.strip() == BLOCK_START)
        i1 = next(i for i, l in enumerate(lines) if l.strip() == BLOCK_END)
    except StopIteration:
        raise ValueError("no OpenVinyl groove block found in file")
    header = {}
    beads = []
    cur = None
    f = None
    for raw in lines[i0 + 1:i1]:
        line = raw.strip()
        if line.startswith("; OV bead="):
            kv = dict(t.split("=") for t in line[5:].split())
            cur = {"name": kv["bead"], "layer": int(kv["layer"]), "dir": kv["dir"],
                   "x": [], "y": [], "f": []}
            beads.append(cur)
            continue
        if line.startswith("; OV "):
            k, v = line[5:].split("=", 1)
            try:
                header[k] = float(v)
            except ValueError:
                header[k] = v
            continue
        if not line.startswith("G1") or cur is None:
            continue
        pr = {k: float(v) for k, v in _PARAM.findall(line.split(";")[0])}
        if "F" in pr:
            f = pr["F"] / 60.0
        if "X" in pr and "Y" in pr and "E" in pr and pr["E"] > 0:
            cur["x"].append(pr["X"])
            cur["y"].append(pr["Y"])
            cur["f"].append(f)
        elif "X" in pr and "Y" in pr and "E" not in pr and not cur["x"]:
            cur["x"].append(pr["X"])   # travel to first vertex
            cur["y"].append(pr["Y"])
            cur["f"].append(f)
    for b in beads:
        for k in ("x", "y", "f"):
            b[k] = np.asarray(b[k], dtype=float)
    return {"header": header, "beads": beads}


def _foot_on_polyline(px, py, qx, qy):
    """For each point p, the closest point on the polyline q (segment projection)."""
    from scipy.spatial import cKDTree
    j = cKDTree(np.column_stack([qx, qy])).query(np.column_stack([px, py]))[1]
    best = np.full(len(px), np.inf)
    fx, fy = qx[j].astype(float), qy[j].astype(float)
    for a_off in (-1, 0):
        a = np.clip(j + a_off, 0, len(qx) - 2)
        ax, ay = qx[a], qy[a]
        dx, dy = qx[a + 1] - ax, qy[a + 1] - ay
        L2 = dx * dx + dy * dy
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.clip(np.where(L2 > 0, ((px - ax) * dx + (py - ay) * dy) / L2, 0.0), 0, 1)
        cx_, cy_ = ax + u * dx, ay + u * dy
        d = np.hypot(px - cx_, py - cy_)
        m = d < best
        best[m], fx[m], fy[m] = d[m], cx_[m], cy_[m]
    return fx, fy


def _polar(b, cx, cy):
    x, y = b["x"] - cx, b["y"] - cy
    r = np.hypot(x, y)
    th = np.unwrap(np.arctan2(y, x))
    f = b["f"]
    if b["dir"] == "out":
        r, th, f = r[::-1], th[::-1], f[::-1]
    th = th - th[0] + (th[0] % (2 * math.pi))   # keep the fractional start angle
    if th[0] > math.pi:
        th -= 2 * math.pi
    return r, th, f


def decode(parsed: dict, fs: int = 44100, playback_model: bool = True) -> dict:
    h = parsed["header"]
    cx, cy = h["center_x"], h["center_y"]
    k = h["pitch"] / (2 * math.pi)
    omega = 2 * math.pi * h["rpm"] / 60.0
    L = next(b for b in parsed["beads"] if b["name"] == "L" and b["layer"] == 0)
    R = next(b for b in parsed["beads"] if b["name"] == "R" and b["layer"] == 0)
    rL, thL, fL = _polar(L, cx, cy)
    rR, thR, _ = _polar(R, cx, cy)
    rR_i = np.interp(thL, thR, rR)
    # Crease = midpoint across the groove: L vertex i and R vertex i straddle
    # centreline vertex i along its normal (foot on R as a fallback).
    Lx, Ly = rL * np.cos(thL), rL * np.sin(thL)
    Rx, Ry = rR * np.cos(thR), rR * np.sin(thR)
    if len(Lx) == len(Rx):
        fx, fy = Rx, Ry                      # emitter pairs vertex i of L and R
    else:
        fx, fy = _foot_on_polyline(Lx, Ly, Rx, Ry)
    mx, my = 0.5 * (Lx + fx), 0.5 * (Ly + fy)
    th = np.unwrap(np.arctan2(my, mx))
    th += 2 * math.pi * np.round((thL[0] - th[0]) / (2 * math.pi))
    r_spiral = h["outer_radius"] - k * th
    x = np.hypot(mx, my) - r_spiral
    t = th / omega
    ds = h["segment_length"]

    x_model = x.copy()
    if playback_model:
        win = max(1, int(round(h["extrusion_width"] / ds)))
        if win > 1:
            from scipy.ndimage import uniform_filter1d
            x_model = uniform_filter1d(x_model, size=win, mode="nearest")
    n = int(t[-1] * fs)
    tt = np.arange(n) / fs
    audio = np.interp(tt, t, x_model)
    if playback_model:
        v_inner = omega * h["inner_radius"]
        f_stylus = v_inner / (2 * math.pi * h["stylus_radius"])
        audio = lowpass(audio, fs, f_stylus, order=2)
    peak = np.abs(audio).max()
    audio_n = audio / peak if peak > 0 else audio
    return {"theta": th, "t": t, "x": x, "r_spiral": r_spiral, "rL": rL, "rR": rR_i,
            "feed": fL, "audio": audio_n, "audio_mm": audio, "fs": fs,
            "recovered_amplitude_mm": float(np.abs(x).max()), "header": h}


def render_png(dec: dict, parsed: dict, out_png: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    h = dec["header"]
    cx, cy = h["center_x"], h["center_y"]
    th, rL, rR = dec["theta"], dec["rL"], dec["rR"]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    step = max(1, len(th) // 60000)
    ax.plot((rL * np.cos(th))[::step], (rL * np.sin(th))[::step], lw=0.3, color="k")
    rec = plt.Circle((0, 0), h.get("outer_radius", 120) + 7, fill=False, ls="--", lw=0.8)
    ax.add_patch(rec)
    ax.set_aspect("equal")
    ax.set_title("toolpath (bead L, decimated)")
    ax.set_xlabel("mm")

    ax = axes[0, 1]
    mid = len(th) // 2
    r0 = 0.5 * (rL[mid] + rR[mid])
    th0 = th[mid]
    sel = (th > th0 - 2.5 * 2 * math.pi) & (th < th0 + 0.5 * 2 * math.pi)
    win = 4.0
    for r, c in ((rL, "tab:blue"), (rR, "tab:red")):
        xx, yy = r[sel] * np.cos(th[sel]), r[sel] * np.sin(th[sel])
        x0, y0 = r0 * math.cos(th0), r0 * math.sin(th0)
        m = (np.abs(xx - x0) < win) & (np.abs(yy - y0) < win)
        ax.plot(xx[m], yy[m], ".", ms=1.5, color=c)
    ax.set_aspect("equal")
    ax.set_title(f"beads L/R near r={r0:.1f} mm, {win*2:.0f} mm window (vertices)")

    ax = axes[1, 0]
    t = dec["t"]
    tm = t[mid]
    m = (t > tm) & (t < tm + 0.03)
    ax.plot((t[m] - tm) * 1000, dec["x"][m] * 1000, lw=0.8, label="crease offset (G-code)")
    ax.set_xlabel("ms")
    ax.set_ylabel("um")
    ax.set_title("recovered lateral modulation, 30 ms at mid-record")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1, 1]
    feed = dec["feed"].copy()
    if len(feed) > 1:
        feed[0] = feed[1]          # first vertex carries the travel feedrate
    ax.plot(th / (2 * math.pi), feed, lw=0.5)
    ax.set_xlabel("turn")
    ax.set_ylabel("mm/s")
    ax.set_title("scheduled feedrate along the groove")
    fig.suptitle(f"OpenVinyl preview: {h['rpm']:.0f} RPM, A={h['amplitude']} mm, "
                 f"pitch={h['pitch']:.3f} mm, play {h.get('play_time_s')} s")
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def preview_file(gcode_path: str, out_png: str = None, out_wav: str = None,
                 fs: int = 44100, playback_model: bool = True) -> dict:
    with open(gcode_path, "r", errors="replace") as fh:
        lines = fh.read().splitlines()
    parsed = parse_block(lines)
    dec = decode(parsed, fs=fs, playback_model=playback_model)
    if out_wav:
        write_wav(out_wav, dec["audio"], fs)
    if out_png:
        render_png(dec, parsed, out_png)
    return dec
