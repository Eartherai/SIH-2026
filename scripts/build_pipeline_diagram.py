#!/usr/bin/env python3
"""Regenerate the pipeline diagram used in the SIH slide deck.

The nine stages and their order match the actual pipeline (see
docs/SIH_SLIDES.md / docs/ARCHITECTURE.md) -- this replaces a generic
placeholder graphic that was missing the detection stage entirely.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.path import Path as MPath

OUT = Path(__file__).resolve().parents[1] / "docs" / "images" / "pipeline_diagram_sih.png"

NAVY = "#1B2A4A"
STAGES_ROW1 = [("Raw Sonar", "#CFE3FA", "#2F6FB0"),
               ("Quality Control", "#E1D8F7", "#7A4FC4"),
               ("Preprocessing &\nTiling\n(off by default)", "#D6F0DC", "#3C8C4E")]
STAGES_ROW2 = [("Calibration", "#FBD8D6", "#C24C42"),
               ("Learned FP\nFilter", "#FCE9C2", "#C98A1E"),
               ("Detection\n(YOLO11n)", "#D9E4FB", "#3355A8")]
STAGES_ROW3 = [("Deduplication", "#CFEFEC", "#1C8A80"),
               ("Geolocation\n(or refuse)", "#DAD9F8", "#4B47B0"),
               ("Priority &\nReport", "#E7D9F5", "#7B3FA0")]

fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
ax.set_xlim(0, 12)
ax.set_ylim(0, 8)
ax.axis("off")
fig.patch.set_alpha(0)

BOX_W, BOX_H = 3.15, 1.55
COL_X = [0.5, 4.35, 8.2]
ROW_Y = [6.1, 3.55, 1.0]


def draw_box(x, y, label, fill, edge):
    shadow = FancyBboxPatch((x + 0.045, y - 0.045), BOX_W, BOX_H,
                             boxstyle="round,pad=0.02,rounding_size=0.14",
                             linewidth=0, facecolor="#000000", alpha=0.10, zorder=2.5)
    ax.add_patch(shadow)
    box = FancyBboxPatch((x, y), BOX_W, BOX_H,
                          boxstyle="round,pad=0.02,rounding_size=0.14",
                          linewidth=2.4, edgecolor=edge, facecolor=fill, zorder=3)
    ax.add_patch(box)
    ax.text(x + BOX_W / 2, y + BOX_H / 2, label, ha="center", va="center",
             fontsize=14, fontweight="bold", color=NAVY, zorder=4, linespacing=1.35)


def arrow(x1, y1, x2, y2, color=NAVY):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22,
                         linewidth=2.4, color=color, zorder=2, shrinkA=0, shrinkB=0)
    ax.add_patch(a)


centers = []
for row_boxes, y in zip([STAGES_ROW1, STAGES_ROW2, STAGES_ROW3], ROW_Y):
    row_centers = []
    for x, (label, fill, edge) in zip(COL_X, row_boxes):
        draw_box(x, y, label, fill, edge)
        row_centers.append((x + BOX_W / 2, y + BOX_H / 2, x, y))
    centers.append(row_centers)

# row1: left -> right
for i in range(2):
    cx, cy, x, y = centers[0][i]
    cx2, cy2, x2, y2 = centers[0][i + 1]
    arrow(x + BOX_W, cy, x2 - 0.06, cy2)

# row1[2] (top-right) drops to row2[2] (Detection, right column) -- reading order left->right
# so physically place row2 as Detection, FP Filter, Calibration (entry at right)
# NOTE: STAGES_ROW2 above is ordered [Calibration, FP Filter, Detection] so that the
# flow travels row1-end -> Detection (col3) -> FP Filter (col2) -> Calibration (col1)
cx3, cy3, x3, y3 = centers[0][2]
cx_det, cy_det, x_det, y_det = centers[1][2]
arrow(cx3, y3, cx_det, y_det + BOX_H)

# row2: right -> left  (Detection -> FP Filter -> Calibration)
for i in [2, 1]:
    cx, cy, x, y = centers[1][i]
    cx2, cy2, x2, y2 = centers[1][i - 1]
    arrow(x, cy, x2 + BOX_W + 0.06, cy2)

# row2[0] (Calibration, left) drops to row3[0] (Deduplication, left)
cx_cal, cy_cal, x_cal, y_cal = centers[1][0]
cx_dedup, cy_dedup, x_dedup, y_dedup = centers[2][0]
arrow(cx_cal, y_cal, cx_dedup, y_dedup + BOX_H)

# row3: left -> right
for i in range(2):
    cx, cy, x, y = centers[2][i]
    cx2, cy2, x2, y2 = centers[2][i + 1]
    arrow(x + BOX_W, cy, x2 - 0.06, cy2)

plt.tight_layout(pad=0.3)
fig.savefig(OUT, transparent=True)
print(f"wrote {OUT}")
