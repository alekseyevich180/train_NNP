"""Generate active_learning_workflow_code_aligned as PNG or SVG.

This script recreates the six-module active-learning workflow figure with
editable matplotlib primitives. Modify labels, font sizes, colors, or box
coordinates here, then rerun:

    python generate_active_learning_workflow_code_aligned.py
    python generate_active_learning_workflow_code_aligned.py --output active_learning_workflow_code_aligned.svg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import patches


DEFAULT_OUT = Path(__file__).with_name("active_learning_workflow_code_aligned.png")

W, H = 16.8, 9.5
COLORS = {
    "blue": "#2b74c7",
    "green": "#4a9b50",
    "orange": "#f28e2b",
    "purple": "#7d58b3",
    "pink": "#d84a7a",
    "gray": "#5f6872",
}


def add_round_box(ax, x, y, w, h, edge, face="#ffffff", lw=1.1, r=0.08):
    box = patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.018,rounding_size={r}",
        linewidth=lw,
        edgecolor=edge,
        facecolor=face,
    )
    ax.add_patch(box)
    return box


def add_header(ax, x, y, w, h, color, num, title):
    add_round_box(ax, x, y, w, h, color, "#fbfbfb", lw=1.0, r=0.07)
    ax.add_patch(patches.Circle((x + 0.28, y + h - 0.28), 0.19, color=color))
    ax.text(x + 0.28, y + h - 0.28, str(num), ha="center", va="center",
            color="white", fontsize=16, weight="bold")
    ax.text(x + w / 2 + 0.25, y + h - 0.32, title, ha="center", va="center",
            fontsize=12, weight="bold", linespacing=1.1)


def arrow(ax, x1, y1, x2, y2, color="#4b5563", lw=1.5, ms=13):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, mutation_scale=ms),
    )


def molecule(ax, cx, cy, scale=1.0, surface=False):
    pts = [
        (-0.35, -0.05, "C"),
        (-0.12, 0.18, "C"),
        (0.18, 0.1, "C"),
        (0.38, 0.32, "O"),
        (-0.42, -0.32, "H"),
        (0.08, -0.25, "H"),
    ]
    for i, j in [(0, 1), (1, 2), (2, 3), (0, 4), (2, 5)]:
        x1, y1, _ = pts[i]
        x2, y2, _ = pts[j]
        ax.plot([cx + x1 * scale, cx + x2 * scale], [cy + y1 * scale, cy + y2 * scale],
                color="#555", lw=1.1)
    for x, y, e in pts:
        color = {"O": "#e41a1c", "C": "#80624f", "H": "#f5f5f5"}[e]
        ax.add_patch(patches.Circle((cx + x * scale, cy + y * scale), 0.055 * scale,
                                    facecolor=color, edgecolor="#555", lw=0.8))
    if surface:
        for k in range(5):
            ax.add_patch(patches.Circle((cx - 0.42 * scale + k * 0.22 * scale, cy - 0.55 * scale),
                                        0.07 * scale, facecolor="#7f8f82", edgecolor="#555", lw=0.8))
            ax.add_patch(patches.Circle((cx - 0.42 * scale + k * 0.22 * scale, cy - 0.42 * scale),
                                        0.06 * scale, facecolor="#e41a1c", edgecolor="#555", lw=0.8))


def surface_grid(ax, x, y, cols=8, rows=2, s=0.08):
    for r in range(rows):
        for c in range(cols):
            cx = x + c * 0.22
            cy = y + r * 0.23
            ax.add_patch(patches.Circle((cx, cy), s, facecolor="#7f8f82", edgecolor="#566", lw=0.8))
            ax.add_patch(patches.Circle((cx, cy + 0.12), s * 0.75,
                                        facecolor="#e41a1c", edgecolor="#7a1111", lw=0.6))


def mini_graph(ax, cx, cy, scale=1.0, event=False):
    nodes = [(-0.35, 0.15), (-0.1, 0.35), (0.18, 0.2), (0.32, -0.12), (0.0, -0.25), (-0.3, -0.15)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
    for i, j in edges:
        ax.plot([cx + nodes[i][0] * scale, cx + nodes[j][0] * scale],
                [cy + nodes[i][1] * scale, cy + nodes[j][1] * scale], color="#555", lw=1.0)
    if event:
        ax.plot([cx - 0.04 * scale, cx + 0.24 * scale], [cy - 0.20 * scale, cy - 0.06 * scale],
                color="#d62728", lw=1.2, ls="--")
        ax.plot([cx - 0.04 * scale, cx + 0.12 * scale], [cy - 0.20 * scale, cy - 0.30 * scale],
                color="#1f5fbf", lw=1.2, ls="--")
    for x, y in nodes:
        ax.add_patch(patches.Circle((cx + x * scale, cy + y * scale), 0.06 * scale,
                                    facecolor="#c9d0c6", edgecolor="#555", lw=0.8))


def draw_module_1(ax, x, y, w, h):
    c = COLORS["blue"]
    add_header(ax, x, y, w, h, c, 1, "AIMD Trajectory\nGeneration")
    ax.text(x + w / 2, y + h - 0.9, "Ab initio molecular dynamics\nat ZnO(10-10) surface",
            ha="center", va="top", fontsize=8.6, style="italic")
    molecule(ax, x + 0.88, y + h - 2.0, 0.72)
    molecule(ax, x + 1.55, y + h - 1.95, 0.46)
    surface_grid(ax, x + 0.3, y + h - 3.35, cols=9, rows=2, s=0.075)
    for dx in [0.95, 1.6, 2.1]:
        arrow(ax, x + dx, y + h - 2.35, x + dx - 0.15, y + h - 3.0, "#d62728", lw=1.0, ms=9)
    ax.text(x + w / 2, y + 2.15, "Trajectory timeline", ha="center", fontsize=8.3)
    xs = [x + 0.3, x + 0.75, x + 1.2, x + 1.95]
    for i, xx in enumerate(xs):
        ax.add_patch(patches.Circle((xx, y + 1.75), 0.06,
                                    color=["#3486cf", "#73b56b", "#f2c12e", "#8055c7"][i]))
        ax.text(xx, y + 1.96, [r"$t_0$", r"$t_1$", r"$t_2$", r"$t_N$"][i],
                ha="center", fontsize=8.2)
    ax.plot([xs[0], xs[-1] + 0.28], [y + 1.75, y + 1.75], color="#111", lw=0.9)
    arrow(ax, xs[-1] + 0.12, y + 1.75, xs[-1] + 0.32, y + 1.75, "#111", lw=0.9, ms=8)
    ax.text(x + w / 2, y + 1.25, "Frame sampling", ha="center", fontsize=8.3)
    for i in range(8):
        ax.add_patch(patches.Circle((x + 0.35 + i * 0.25, y + 0.95), 0.055,
                                    facecolor="#ececec", edgecolor="#777", lw=0.6))
    add_round_box(ax, x + 0.1, y + 0.2, w - 0.2, 0.48, c, "#ffffff", lw=0.8, r=0.06)
    for i, (lab, col) in enumerate([("Zn", "#7f8f82"), ("O", "#e41a1c"), ("C", "#80624f"), ("H", "#f5f5f5")]):
        xx = x + 0.35 + i * 0.55
        ax.add_patch(patches.Circle((xx, y + 0.43), 0.07, facecolor=col, edgecolor="#555", lw=0.6))
        ax.text(xx + 0.15, y + 0.43, lab, va="center", fontsize=7.6)


def draw_module_2(ax, x, y, w, h):
    c = COLORS["green"]
    add_header(ax, x, y, w, h, c, 2, "Bond Change\nDetection")
    ax.text(x + w / 2, y + h - 0.9, "Consecutive frame comparison", ha="center", fontsize=8.4)
    for i, lab in enumerate(["Frame $t$", "Frame $t+1$"]):
        bx = x + 0.14 + i * 1.3
        add_round_box(ax, bx, y + h - 3.0, 1.05, 1.65, "#999", "#ffffff", lw=0.7, r=0.06)
        ax.text(bx + 0.52, y + h - 1.55, lab, ha="center", fontsize=8.0)
        molecule(ax, bx + 0.52, y + h - 2.18, 0.55, surface=True)
    arrow(ax, x + 1.33, y + h - 2.15, x + 1.6, y + h - 2.15, "#555")
    ax.text(x + w / 2, y + h - 3.25, "Bond graph difference", ha="center", fontsize=8.4)
    mini_graph(ax, x + 0.7, y + h - 4.0, 0.95)
    mini_graph(ax, x + 1.9, y + h - 4.0, 0.95, event=True)
    arrow(ax, x + 1.18, y + h - 4.0, x + 1.43, y + h - 4.0, "#555")
    ax.plot([x + 0.35, x + 1.05], [y + 2.2, y + 2.2], color="#1f5fbf", ls="--", lw=1.2)
    ax.text(x + 1.2, y + 2.2, "Formed bonds", va="center", fontsize=8.1)
    ax.plot([x + 0.35, x + 1.05], [y + 1.9, y + 1.9], color="#d62728", ls="--", lw=1.2)
    ax.text(x + 1.2, y + 1.9, "Broken bonds", va="center", fontsize=8.1)
    ax.text(x + w / 2, y + 1.45, "Key bond & interface events", ha="center", fontsize=8.4)
    add_round_box(ax, x + 0.15, y + 0.3, w - 0.3, 1.0, "#aaa", "#ffffff", lw=0.7, r=0.06)
    for i, lab in enumerate(["Bond\nformed", "Bond\nbroken", "Interface\nchange"]):
        cx = x + 0.55 + i * 0.78
        ax.text(cx, y + 0.45, lab, ha="center", va="bottom", fontsize=7.7)
        mini_graph(ax, cx, y + 0.95, 0.35, event=i > 0)


def draw_module_3(ax, x, y, w, h):
    c = COLORS["orange"]
    add_header(ax, x, y, w, h, c, 3, "Zn-O Coordination\nAnalysis")
    ax.text(x + w / 2, y + h - 0.9, "Zn-O coordination number (CN)\nover time",
            ha="center", va="top", fontsize=8.2)
    px, py = x + 0.35, y + h - 3.05
    ax.text(px - 0.18, py + 1.38, "CN", fontsize=8.2)
    ax.plot([px, px], [py, py + 1.25], color="#333", lw=0.9)
    ax.plot([px, px + 1.9], [py, py], color="#333", lw=0.9)
    vals = [2.0, 4.5, 2.2, 2.7, 4.7, 2.8, 1.5]
    xs = [px + i * 0.32 for i in range(len(vals))]
    ys = [py + v / 5.2 * 1.15 for v in vals]
    ax.plot(xs, ys, color="#4b5f7d", lw=1.0)
    for xx, yy, col in zip(xs, ys, ["#3486cf", "#73b56b", "#f2a12e", "#f28e2b", "#e41a1c", "#8c65c6", "#8c65c6"]):
        ax.plot([xx, xx], [py, yy], color="#777", lw=0.8, ls="--")
        ax.add_patch(patches.Circle((xx, yy), 0.06, facecolor=col, edgecolor="#555", lw=0.6))
    ax.text(px + 0.95, py - 0.42, "Time (frames)", ha="center", fontsize=8.0, style="italic")
    ax.text(x + w / 2, y + 3.2, "Event classes", ha="center", fontsize=8.5)
    for cx, sym, col, lab in [
        (x + 0.45, "+", "#3e9b45", "Adsorption\n(CN up)"),
        (x + 1.25, "-", "#e41a1c", "Desorption\n(CN down)"),
        (x + 2.05, "<->", "#1f5fbf", "Coordination\nshift"),
    ]:
        ax.add_patch(patches.Circle((cx, y + 2.55), 0.23, facecolor="white", edgecolor=col, lw=1.2))
        ax.text(cx, y + 2.55, sym, ha="center", va="center", fontsize=24, color=col, weight="bold")
        ax.text(cx, y + 1.82, lab, ha="center", fontsize=8.0)
    add_round_box(ax, x + 0.1, y + 0.25, w - 0.2, 1.1, c, "#ffffff", lw=0.8, r=0.06)
    ax.text(x + w / 2, y + 1.13, "Coordination change\naround Zn sites",
            ha="center", va="top", fontsize=8.2)
    for cx in [x + 0.75, x + 1.85]:
        ax.add_patch(patches.Circle((cx, y + 0.62), 0.11, facecolor="#7f8f82", edgecolor="#555", lw=0.7))
        for dx, dy in [(0.0, 0.35), (0.0, -0.35), (-0.32, 0.0), (0.32, 0.0)]:
            ax.add_patch(patches.Circle((cx + dx, y + 0.62 + dy), 0.06, facecolor="#e41a1c", edgecolor="#7a1111", lw=0.6))
    arrow(ax, x + 1.18, y + 0.62, x + 1.43, y + 0.62, "#666")


def draw_module_4(ax, x, y, w, h):
    c = COLORS["purple"]
    add_header(ax, x, y, w, h, c, 4, "Descriptor and\nvdW Scoring")
    ax.text(x + w / 2, y + h - 0.9, "SOAP descriptors", ha="center", fontsize=8.4)
    add_round_box(ax, x + 0.1, y + h - 1.35, w - 0.2, 0.28, "#777", "#ffffff", lw=0.6, r=0.04)
    for i in range(7):
        ax.add_patch(patches.Rectangle((x + 0.22 + i * 0.22, y + h - 1.27), 0.15, 0.14,
                                       facecolor=plt.cm.Purples(0.75 - i * 0.06), edgecolor="#555", lw=0.4))
    ax.text(x + w - 0.28, y + h - 1.2, "...", ha="center", va="center", fontsize=12)
    add_round_box(ax, x + 0.1, y + h - 3.4, w - 0.2, 1.85, "#aaa", "#ffffff", lw=0.7, r=0.05)
    ax.text(x + w / 2, y + h - 1.78, "Local atomic environments", ha="center", fontsize=7.9)
    molecule(ax, x + 1.2, y + h - 2.9, 0.55)
    for cx, cy, ec in [(x + 0.75, y + h - 2.35, c), (x + 1.25, y + h - 2.95, "#c9874b"), (x + 2.0, y + h - 2.55, "#58a0e8")]:
        ax.add_patch(patches.Ellipse((cx, cy), 0.65, 0.55, fill=False, edgecolor=ec, ls="--", lw=0.8))
    ax.text(x + w / 2, y + h - 3.75, "vdW interaction categories", ha="center", fontsize=8.4)
    rows = [("organic-surface", "#dff3df"), ("water-surface", "#e6f1ff"), ("organic-water", "#fff0e4")]
    for i, (lab, face) in enumerate(rows):
        yy = y + h - 4.35 - i * 0.68
        add_round_box(ax, x + 0.12, yy, w - 0.24, 0.55, ["#4a9b50", "#4b95e6", "#f28e2b"][i], face, lw=0.7, r=0.05)
        molecule(ax, x + 0.55, yy + 0.28, 0.35, surface=True)
        ax.text(x + w - 0.6, yy + 0.28, lab, ha="center", va="center", fontsize=7.5)
    ax.text(x + w / 2, y + 0.5, "vdW scores (heuristic)", ha="center", fontsize=8.2)
    ax.text(x + 0.1, y + 0.27, "Low", fontsize=8.0)
    ax.text(x + w - 0.12, y + 0.27, "High", fontsize=8.0, ha="right")
    ax.imshow([[i for i in range(100)]], extent=(x + 0.48, x + w - 0.48, y + 0.23, y + 0.34),
              cmap="Purples", aspect="auto", zorder=0)


def draw_module_5(ax, x, y, w, h):
    c = COLORS["pink"]
    add_header(ax, x, y, w, h, c, 5, "Neural Network\nStructure Selector (MLP)")
    ax.text(x + w / 2, y + h - 0.95, "Structure selector, not an energy predictor\nNot committee uncertainty",
            ha="center", va="top", fontsize=8.1, color="#cc0033")
    ax.text(x + w / 2, y + h - 1.65, "Multi-source feature fusion", ha="center", fontsize=8.4)
    labels = [("Bond signals", "#eaf6ea"), ("Zn-O coordination\nsignals", "#fff0e4"), ("SOAP features", "#f1edff"), ("vdW scoring cues", "#eaf3ff")]
    for i, (lab, face) in enumerate(labels):
        yy = y + h - 2.5 - i * 0.72
        add_round_box(ax, x + 0.12, yy, 1.05, 0.58, ["#4a9b50", "#f28e2b", "#7d58b3", "#4b95e6"][i], face, lw=0.7, r=0.05)
        ax.text(x + 0.64, yy + 0.45, lab, ha="center", va="top", fontsize=7.0)
        mini_graph(ax, x + 0.64, yy + 0.17, 0.35, event=i < 2)
        arrow(ax, x + 1.17, yy + 0.29, x + 1.45, y + h - 3.5, "#111", lw=0.8, ms=8)
    add_round_box(ax, x + 1.45, y + h - 4.6, 0.45, 3.0, "#777", "#ffffff", lw=0.7, r=0.05)
    ax.text(x + 1.68, y + h - 3.3, "Feature\nfusion", ha="center", va="center", fontsize=7.6)
    for i, col in enumerate(["#c9e6c9", "#ffcf91", "#d8cff0", "#f5bfd0"]):
        ax.add_patch(patches.Rectangle((x + 1.55, y + h - 4.35 + i * 0.25), 0.25, 0.22,
                                       facecolor=col, edgecolor="#888", lw=0.4))
    arrow(ax, x + 1.9, y + h - 3.1, x + 2.23, y + h - 3.1, "#111", lw=0.9, ms=9)
    ax.text(x + 2.72, y + h - 2.38, "MLP scoring model", ha="center", fontsize=7.9)
    add_round_box(ax, x + 2.23, y + h - 4.0, 1.05, 1.25, "#999", "#ffffff", lw=0.7, r=0.04)
    for i in range(5):
        ax.add_patch(patches.Circle((x + 2.35, y + h - 3.75 + i * 0.25), 0.06, facecolor="#a6dba0", edgecolor="#555", lw=0.5))
        ax.add_patch(patches.Circle((x + 2.78, y + h - 3.68 + i * 0.18), 0.07, facecolor="#bdd7ee", edgecolor="#555", lw=0.5))
    ax.add_patch(patches.Circle((x + 3.15, y + h - 3.3), 0.07, facecolor="#fed976", edgecolor="#555", lw=0.5))
    for yy in [y + h - 3.75 + i * 0.25 for i in range(5)]:
        ax.plot([x + 2.41, x + 2.72], [yy, y + h - 3.3], color="#777", lw=0.5)
    for yy in [y + h - 3.68 + i * 0.18 for i in range(5)]:
        ax.plot([x + 2.85, x + 3.08], [yy, y + h - 3.3], color="#777", lw=0.5)
    arrow(ax, x + 2.78, y + h - 4.05, x + 2.78, y + h - 4.4, "#111", lw=0.9, ms=9)
    ax.text(x + 2.78, y + h - 4.62, "Output\nSelection score\n(0-1)", ha="center", va="top", fontsize=7.9)
    ax.imshow([[i for i in range(100)]], extent=(x + 2.35, x + 3.3, y + h - 5.28, y + h - 5.15),
              cmap="RdPu", aspect="auto", zorder=0)
    ax.text(x + 2.25, y + h - 5.28, "0", fontsize=7.7, va="center")
    ax.text(x + 3.36, y + h - 5.28, "1", fontsize=7.7, va="center")
    ax.text(x + w / 2, y + 1.0, "High-value candidates", ha="center", fontsize=8.4)
    for i in range(4):
        add_round_box(ax, x + 0.12 + i * 0.8, y + 0.15, 0.7, 0.75, "#aaa", "#ffffff", lw=0.6, r=0.04)
        molecule(ax, x + 0.47 + i * 0.8, y + 0.48, 0.4, surface=True)
    ax.text(x + w - 0.28, y + 0.55, "...", fontsize=16, ha="center")


def draw_module_6(ax, x, y, w, h):
    c = COLORS["blue"]
    add_header(ax, x, y, w, h, c, 6, "Final Training\nDataset Assembly")
    ax.text(x + w / 2, y + h - 0.9, "Candidate aggregation", ha="center", fontsize=8.2)
    ax.add_patch(patches.Ellipse((x + w / 2, y + h - 1.65), 1.45, 0.7, fill=False, edgecolor="#aaa", ls="--", lw=0.8))
    for i, col in enumerate(["#a6dba0", "#f6a6b8", "#bdd7ee", "#fed976"] * 3):
        xx = x + 0.45 + (i % 6) * 0.25
        yy = y + h - 1.9 + (i // 6) * 0.25
        ax.add_patch(patches.Circle((xx, yy), 0.06, facecolor=col, edgecolor="#777", lw=0.5))
    for yy, lab in [(y + h - 2.45, "Deduplication"), (y + h - 3.65, "Provenance tracking"), (y + h - 4.85, "Dataset split")]:
        arrow(ax, x + w / 2, yy + 0.45, x + w / 2, yy + 0.1, "#555")
        ax.text(x + w / 2, yy - 0.05, lab, ha="center", fontsize=8.2)
    for i, col in enumerate(["#dbe5f1", "#eaf2f8", "#e8f5e9", "#fef2cc"]):
        ax.add_patch(patches.Circle((x + 0.65 + i * 0.34, y + h - 3.2), 0.33,
                                    fill=False, edgecolor=["#5b6f99", "#558fa6", "#7aa95c", "#e0a13a"][i], lw=0.9))
    ax.plot([x + 0.35, x + w - 0.25], [y + h - 4.05, y + h - 4.05], color="#888", lw=0.8)
    for i, col in enumerate(["#3486cf", "#73b56b", "#f2c12e", "#f28e2b", "#8055c7"]):
        xx = x + 0.35 + i * 0.42
        ax.add_patch(patches.Circle((xx, y + h - 3.7), 0.06, facecolor=col, edgecolor="#555", lw=0.5))
        ax.plot([xx, xx], [y + h - 3.76, y + h - 4.05], color="#777", lw=0.7, ls="--")
    ax.pie([80, 10, 10], colors=["#cfe8cf", "#b7d3f2", "#ffd08a"],
           radius=0.42, center=(x + 1.0, y + 0.85),
           wedgeprops=dict(edgecolor="#3a6f3c", linewidth=0.6))
    ax.text(x + 0.2, y + 0.55, "Train\n(~80%)", ha="left", fontsize=8.1, color="#188038")
    ax.text(x + 1.55, y + 0.45, "Test\n(~10%)", ha="left", fontsize=8.1, color="#0b57d0")
    ax.text(x + 1.75, y + 0.98, "Val\n(~10%)", ha="left", fontsize=8.1, color="#e8710a")


def draw_core_logic(ax):
    y, h = 0.35, 1.55
    add_round_box(ax, 0.1, y, 16.6, h, COLORS["blue"], "#ffffff", lw=1.0, r=0.08)
    ax.add_patch(patches.FancyBboxPatch((0.1, y), 0.95, h, boxstyle="round,pad=0.018,rounding_size=0.08",
                                        facecolor="#0b5da8", edgecolor="#0b5da8"))
    ax.text(0.58, y + h / 2, "Core\nLogic", color="white", ha="center", va="center",
            fontsize=16, weight="bold", linespacing=1.25)
    items = [
        (1.25, 3.6, COLORS["green"], "Chemical event detection",
         "Identify bond and coordination\nchanges that capture reactive\nand interfacial events."),
        (5.15, 3.1, COLORS["orange"], "Feature-based ranking",
         "Quantify structures using\nSOAP descriptors and vdW\ninteraction cues."),
        (8.55, 3.65, COLORS["pink"], "MLP structure selector",
         "Fuse multi-source features to score\nstructures and prioritize high-value\ncandidates for labeling."),
        (12.45, 3.75, COLORS["blue"], "Final data assembly",
         "Aggregate, deduplicate, track provenance,\nand split into train/val/test sets for\nNNP training."),
    ]
    for x, w, c, title, desc in items:
        add_round_box(ax, x, y + 0.2, w, h - 0.4, c, "#ffffff", lw=0.9, r=0.06)
        if "Chemical" in title:
            mini_graph(ax, x + 0.65, y + 0.78, 0.8)
        elif "Feature" in title:
            for i, col in enumerate(["#8fa3c7", "#8fcf8f", "#8bb8e8", "#f0a444"]):
                ax.add_patch(patches.Rectangle((x + 0.32 + i * 0.18, y + 0.38), 0.11, 0.28 + 0.14 * i,
                                               facecolor=col, edgecolor="#555", lw=0.5))
            ax.plot([x + 0.22, x + 1.05], [y + 0.38, y + 0.38], color="#555", lw=0.7)
            ax.plot([x + 0.22, x + 0.22], [y + 0.38, y + 1.2], color="#555", lw=0.7)
        elif "MLP" in title:
            mini_graph(ax, x + 0.55, y + 0.82, 0.75, event=True)
        else:
            for dx, dy, col in [(0.0, 0.12, "#f5b5c8"), (0.22, 0.12, "#9fc5e8"),
                                (0.0, -0.12, "#f5b5c8"), (0.22, -0.12, "#9fc5e8")]:
                add_round_box(ax, x + 0.38 + dx, y + 0.7 + dy, 0.22, 0.22, "#555", col, lw=0.5, r=0.04)
        ax.text(x + 1.45, y + 1.05, title, ha="left", va="center", fontsize=11, weight="bold")
        ax.text(x + 1.45, y + 0.72, desc, ha="left", va="center", fontsize=8.4, linespacing=1.3)


def build_figure():
    fig, ax = plt.subplots(figsize=(W, H), dpi=160)
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(W / 2, H - 0.15,
            "Active Learning Workflow for Neural Network Potential Data Selection from AIMD Trajectories",
            ha="center", va="top", fontsize=18, weight="bold")

    top_y, top_h = 2.35, 6.35
    widths = [2.55, 2.45, 2.45, 2.55, 3.55, 2.05]
    gap = 0.22
    xs = [0.1]
    for w in widths[:-1]:
        xs.append(xs[-1] + w + gap)
    modules = [
        (draw_module_1, COLORS["blue"]),
        (draw_module_2, COLORS["green"]),
        (draw_module_3, COLORS["orange"]),
        (draw_module_4, COLORS["purple"]),
        (draw_module_5, COLORS["pink"]),
        (draw_module_6, COLORS["blue"]),
    ]
    for fn, color, x, w in zip([m[0] for m in modules], [m[1] for m in modules], xs, widths):
        add_round_box(ax, x, top_y, w, top_h, color, "#ffffff", lw=1.0, r=0.07)
        fn(ax, x, top_y, w, top_h)
    for i in range(len(xs) - 1):
        arrow(ax, xs[i] + widths[i] + 0.04, top_y + top_h / 2, xs[i + 1] - 0.04, top_y + top_h / 2,
              "#5b6470", lw=1.6, ms=14)

    draw_core_logic(ax)

    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output PNG path. Default: {DEFAULT_OUT.name}",
    )
    args = parser.parse_args()

    fig = build_figure()
    fig.savefig(args.output, dpi=160, bbox_inches="tight", pad_inches=0.03)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
