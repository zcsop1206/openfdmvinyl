"""
build_figures.py - CAD figures for docs/BUILD.md, rendered from spine_assembly.py.

    python build_figures.py      # writes docs/figures/build_assembly.png, build_pivot.png, build_top.png

A small depth-buffered renderer (orthographic, flat shading, outlines where parts or depth change), so it needs only
numpy and matplotlib beyond build123d. The tessellated assembly is cached in out/build_figures.npz; delete that file
after changing the CAD.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
FIG = ROOT / "docs" / "figures"
CACHE = ROOT / "out" / "build_figures.npz"

COLOUR = {"record 7 in": "#3a3a3a", "cartridge (CKANDAY)": "#c4502f", "T-pin": "#c8c8c8", "pivot screw": "#8f8f8f",
          "clamp screw": "#8f8f8f", "servo STS3215": "#4a4a4a", "horn": "#d8d8d8", "shims": "#c8c8c8"}
GROUP = {"frame": "#aab2bd", "platter": "#e6e3dc", "arm": "#2a5ea8"}
LIGHT = np.array([-0.35, -0.5, 0.8]) / np.linalg.norm([-0.35, -0.5, 0.8])
INK = "#222222"

plt.rcParams.update({"font.family": "Times New Roman", "font.size": 12})


def load():
    if CACHE.exists():
        return np.load(CACHE, allow_pickle=True)["parts"].item()
    import spine_assembly as SA
    parts = {}
    for name, (shape, group) in SA.assembly().items():
        v, f = shape.tessellate(0.15, 0.3)
        v = np.array([[p.X, p.Y, p.Z] for p in v])
        parts[name] = (v[np.array(f)], group)
    CACHE.parent.mkdir(exist_ok=True)
    np.savez_compressed(CACHE, parts=np.array(parts, dtype=object))
    return parts


def clip(t, axis, val, keep_above):
    """Clip triangles (n, 3, 3) to one side of the plane x[axis] = val; returns whole and cut triangles."""
    d = (t[..., axis] - val) * (1 if keep_above else -1)
    inside = d >= 0
    part = inside.any(1) & ~inside.all(1)
    out = [t[inside.all(1)]]
    for tri, dd, ins in zip(t[part], d[part], inside[part]):
        poly = []
        for i in range(3):
            j = (i + 1) % 3
            if ins[i]:
                poly.append(tri[i])
            if ins[i] != ins[j]:
                poly.append(tri[i] + (tri[j] - tri[i]) * dd[i] / (dd[i] - dd[j]))
        for k in range(1, len(poly) - 1):
            out.append(np.array([[poly[0], poly[k], poly[k + 1]]]))
    return np.concatenate(out)


def camera(azim, elev):
    """Rows: screen right, screen up, toward the viewer (world z up)."""
    a, e = np.radians(azim), np.radians(elev)
    toward = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    up0 = np.array([0, 0, 1.0]) if abs(e) < np.radians(89) else np.array([-np.cos(a), -np.sin(a), 0.0])
    right = np.cross(up0, toward); right /= np.linalg.norm(right)
    return np.array([right, np.cross(toward, right), toward])


class View:
    def __init__(self, parts, azim, elev, px=1600, keep=None, box=None, pad=0.04):
        self.R = camera(azim, elev)
        self.ids = {name: k + 1 for k, name in enumerate(parts)}
        tris, cols, ids = [], [], []
        for k, (name, (t, g)) in enumerate(parts.items()):
            if keep and name not in keep:
                continue
            if box is not None:
                for ax_ in range(3):
                    t = clip(t, ax_, box[0][ax_], True) if len(t) else t
                    t = clip(t, ax_, box[1][ax_], False) if len(t) else t
            if not len(t):
                continue
            n = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            rgb = np.array(matplotlib.colors.to_rgb(COLOUR.get(name, GROUP[g])))
            lam = np.abs(n @ LIGHT)
            tris.append(t); cols.append(np.clip(rgb * (0.5 + 0.55 * lam[:, None]), 0, 1)); ids.append(np.full(len(t), k + 1))
        T = np.concatenate(tris); C = np.concatenate(cols); I = np.concatenate(ids)
        P = T @ self.R.T
        lo, hi = P.reshape(-1, 3).min(0), P.reshape(-1, 3).max(0)
        span = (hi - lo)[:2].max() * (1 + 2 * pad)
        self.s = px / span
        self.W = int(np.ceil((hi[0] - lo[0]) * self.s + 2 * pad * span * self.s))
        self.H = int(np.ceil((hi[1] - lo[1]) * self.s + 2 * pad * span * self.s))
        self.o = np.array([lo[0] - pad * span, hi[1] + pad * span])
        self.image(P, C, I)

    def to_px(self, xyz):
        p = np.asarray(xyz, float) @ self.R.T
        return np.array([(p[0] - self.o[0]) * self.s, (self.o[1] - p[1]) * self.s])

    def image(self, P, C, I):
        W, H = self.W, self.H
        z = np.full((H, W), -np.inf); img = np.ones((H, W, 3)); pid = np.zeros((H, W), int)
        X = (P[..., 0] - self.o[0]) * self.s; Y = (self.o[1] - P[..., 1]) * self.s; Z = P[..., 2]
        for k in range(len(P)):
            x, y, d = X[k], Y[k], Z[k]
            x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())), W - 1)
            y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())), H - 1)
            if x1 < x0 or y1 < y0:
                continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
            den = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(den) < 1e-12:
                continue
            a = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / den
            b = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / den
            c = 1 - a - b
            m = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
            if not m.any():
                continue
            dz = a * d[0] + b * d[1] + c * d[2]
            sub = z[y0:y1 + 1, x0:x1 + 1]
            w = m & (dz > sub)
            sub[w] = dz[w]
            img[y0:y1 + 1, x0:x1 + 1][w] = C[k]
            pid[y0:y1 + 1, x0:x1 + 1][w] = I[k]
        # outlines: a change of part, or a jump in depth inside a part
        e = np.zeros((H, W), bool)
        for sl0, sl1 in (((slice(1, None), slice(None)), (slice(None, -1), slice(None))),
                         ((slice(None), slice(1, None)), (slice(None), slice(None, -1)))):
            jump = (pid[sl0] != pid[sl1]) | (np.abs(np.nan_to_num(z[sl0] - z[sl1], posinf=1e9, neginf=1e9)) > 1.2)
            e[sl0] |= jump
        e &= (pid > 0) | np.roll(pid > 0, 1, 0) | np.roll(pid > 0, 1, 1)
        img[e] = np.array(matplotlib.colors.to_rgb(INK)) * 0.9 + img[e] * 0.1
        self.img = img; self.pid = pid

    def anchor(self, xyz, part=None):
        """The projected point, moved to the nearest pixel where `part` is visible (so a leader never lands on
        whatever hides it)."""
        p = self.to_px(xyz)
        if part is None:
            return p
        yy, xx = np.nonzero(self.pid == self.ids[part])
        k = np.argmin((xx + 0.5 - p[0]) ** 2 + (yy + 0.5 - p[1]) ** 2)
        return np.array([xx[k] + 0.5, yy[k] + 0.5])


def figure(view, labels, w_in=10, margin_l=2.6, margin_r=2.6, top=0.4, bottom=0.4, note=None, dpi=200):
    """The render in the middle, labels in the side margins, short leaders, no crossings within a side."""
    ar = view.H / view.W
    img_w = w_in - margin_l - margin_r
    h_in = img_w * ar + top + bottom
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = fig.add_axes((margin_l / w_in, bottom / h_in, img_w / w_in, img_w * ar / h_in))
    ax.imshow(view.img, interpolation="lanczos"); ax.set_axis_off()
    ax.set_xlim(0, view.W); ax.set_ylim(view.H, 0)
    def fig_xy(p):
        return fig.transFigure.inverted().transform(ax.transData.transform(p))
    sides = {"l": [], "r": []}
    for lab in labels:
        xyz, text, side = lab[:3]
        part = lab[3] if len(lab) > 3 else None
        yt = lab[4] if len(lab) > 4 else None             # a fixed text height (figure fraction), to untangle leaders
        sides[side].append((fig_xy(view.anchor(xyz, part)), text, yt))
    gap = 0.42 / h_in * 1.0
    for side, items in sides.items():
        if items and all(it[2] is not None for it in items):
            x_text = (margin_l - 0.15) / w_in if side == "l" else 1 - (margin_r - 0.15) / w_in
            for p, text, yt in items:
                ax.annotate(text, xy=p, xycoords="figure fraction", xytext=(x_text, yt), textcoords="figure fraction",
                            ha="right" if side == "l" else "left", va="center", color=INK,
                            arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=3, shrinkB=0))
                fig.add_artist(matplotlib.patches.Circle(p, 0.0028, transform=fig.transFigure, color=INK))
            continue
        items = [it[:2] for it in items]
        items.sort(key=lambda t: -t[0][1])
        ys = [it[0][1] for it in items]
        for i in range(1, len(ys)):                       # push down to keep a minimum gap
            ys[i] = min(ys[i], ys[i - 1] - gap)
        over = (bottom / h_in * 0.5) - ys[-1] if ys else 0
        if over > 0:
            ys = [y + over for y in ys]
            for i in range(len(ys) - 2, -1, -1):
                ys[i] = max(ys[i], ys[i + 1] + gap)
        x_text = (margin_l - 0.15) / w_in if side == "l" else 1 - (margin_r - 0.15) / w_in
        for (p, text), y in zip(items, ys):
            ax.annotate(text, xy=p, xycoords="figure fraction", xytext=(x_text, y), textcoords="figure fraction",
                        ha="right" if side == "l" else "left", va="center", color=INK,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color=INK, shrinkA=3, shrinkB=0))
            fig.add_artist(matplotlib.patches.Circle(p, 0.0028, transform=fig.transFigure, color=INK))
    if note:
        fig.text(0.5, 0.25 / h_in, note, ha="center", va="center", fontsize=10.5, color="#555555")
    return fig, ax, fig_xy


def centre(parts, name):
    t = parts[name][0].reshape(-1, 3)
    return (t.min(0) + t.max(0)) / 2


def assembly_fig(P):
    v = View(P, azim=-62, elev=26, px=1700)
    # (point, text, side, part the leader lands on, text height); heights set by hand so no two leaders cross
    L = [((-55, 40, 66.1), "record, 7 in", "l", "record 7 in", 0.84),
         ((-89, 20, 64), "clamp ring on the record's edge", "l", "clamp ring", 0.76),
         ((0, 0, 74.1), "centre weight", "l", "centre weight", 0.66),
         ((-60, -70, 56), "platter", "l", "platter (open)", 0.55),
         ((14, -70, 72), "headshell and CZ-800 cartridge", "l", "cartridge (CKANDAY)", 0.43),
         ((-5, -45, 30), "servo pocket (the servo turns the platter)", "l", "spine", 0.30),
         ((60, -30, 95.5), "arm (unipivot)", "r", "arm", 0.73),
         ((120, 0, 96), "pivot: T-pin point in an M3 cup", "r", "pivot screw", 0.65),
         ((150, 25, 80), "counterweight carriage", "r", "weight", 0.55),
         ((131, 0, 74), "damper cup and skirt", "r", "damper cup", 0.43),
         ((75, 0, 8), "spine: servo pocket, beam, pivot tower", "r", "spine", 0.24)]
    fig, *_ = figure(v, L, w_in=11, margin_l=2.9, margin_r=2.9)
    fig.savefig(FIG / "build_assembly.png", facecolor="white"); plt.close(fig)


def pivot_fig(P):
    box = (np.array([100, 0.0, 38]), np.array([140, 40, 100]))    # cut on y = 0: the half toward the viewer removed
    keep = ["spine", "peg", "T-pin", "shims", "pivot screw", "damper cup", "clamp screw", "damper skirt", "arm"]
    v = View(P, azim=-90, elev=8, px=1300, keep=keep, box=box)
    L = [((120, 0, 90.5), "T-pin, point up", "r"),
         ((119.8, 0, 95), "M3 x 8 screw: the pivot cup", "l"),
         ((115.5, 0, 62), "sleeve and tower, clamped to the pin", "l"),
         ((119.8, 4, 60), "M3 x 12 clamp screw", "r"),
         ((131, 0, 72), "damper cup: dish soap or honey", "r"),
         ((112, 0, 76), "skirt on the arm's hub", "l"),
         ((120, 0, 44.2), "shims set the pin's height", "l")]
    fig, *_ = figure(v, L, w_in=10, margin_l=3.0, margin_r=2.7, bottom=0.55,
                     note="section through the pivot axis, seen from the front")
    fig.savefig(FIG / "build_pivot.png", facecolor="white"); plt.close(fig)


def top_fig(P):
    v = View(P, azim=-90, elev=90, px=1500)
    fig, ax, fig_xy = figure(v, [((120, 0, 90), "pivot", "r"), ((150, 25, 80), "counterweight", "r"),
                                 ((0, 0, 75), "centre weight, servo below", "l")],
                             w_in=10, margin_l=2.4, margin_r=2.4)
    red, green = "#c4502f", "#2f8f68"
    th = np.linspace(np.radians(160), np.radians(70), 60)                 # decreasing angle seen from above: clockwise
    pts = np.array([fig_xy(v.to_px((70 * np.cos(t), 70 * np.sin(t), 70))) for t in th])
    fig.add_artist(matplotlib.lines.Line2D(pts[:-2, 0], pts[:-2, 1], transform=fig.transFigure, color=red, lw=1.8))
    fig.add_artist(matplotlib.patches.FancyArrowPatch(pts[-6], pts[-1], transform=fig.transFigure, arrowstyle="-|>",
                                                      mutation_scale=16, color=red, lw=1.8))
    fig.text(0.05, 0.88, "platter turns clockwise", color=red, ha="left", va="center", fontsize=12.5)
    s = centre(P, "cartridge (CKANDAY)")
    d = np.array([s[1], -s[0], 0]); d /= np.linalg.norm(d)                # the disc's motion under the stylus
    a0, a1 = fig_xy(v.to_px(s + 24 * d)), fig_xy(v.to_px(s + 58 * d))
    fig.add_artist(matplotlib.patches.FancyArrowPatch(a0, a1, transform=fig.transFigure, arrowstyle="-|>",
                                                      mutation_scale=16, color=green, lw=1.8, linestyle=(0, (4, 2))))
    fig.text(0.05, 0.12, "under the stylus the groove runs\nfrom the cartridge's pins to its tip",
             color=green, ha="left", va="center", fontsize=12)
    fig.savefig(FIG / "build_top.png", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    parts = load()
    assembly_fig(parts); pivot_fig(parts); top_fig(parts)
    print("wrote", ", ".join(f"docs/figures/{n}" for n in ("build_assembly.png", "build_pivot.png", "build_top.png")))
