#!/usr/bin/env python3
"""Figure 5: theory-guided MCRE measurement and causal-estimation map."""

# Portable publication paths; no experiment parameters are changed.
import sys as _dcml_sys
from pathlib import Path as _DCMLPath
_dcml_sys.path.insert(0, str(_DCMLPath(__file__).resolve().parents[1]))
import project_paths as _dcml_paths


from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = _dcml_paths.workspace_path(_dcml_paths.workspace_path(ROOT, "causal"), "Unified_Visualization/output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

COLORS = {
    "ink": "#20252B",
    "muted": "#5D6670",
    "line": "#8A949D",
    "input": "#E9EEF2",
    "mnc": "#F5D9D2",
    "mnc_dark": "#A84F3D",
    "mai": "#D8E8E2",
    "mai_dark": "#2D7366",
    "output": "#E3E2F0",
    "output_dark": "#565184",
    "white": "#FFFFFF",
}


def box(
    ax,
    x,
    y,
    w,
    h,
    face,
    edge,
    title,
    body="",
    title_color=None,
    lw=1.2,
    body_size=8.4,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        facecolor=face,
        edgecolor=edge,
        linewidth=lw,
    )
    ax.add_patch(patch)
    ax.text(
        x + 0.16,
        y + h - 0.18,
        title,
        ha="left",
        va="top",
        fontsize=10.6,
        fontweight="bold",
        color=title_color or COLORS["ink"],
    )
    if body:
        ax.text(
            x + 0.16,
            y + h - 0.55,
            body,
            ha="left",
            va="top",
            fontsize=body_size,
            linespacing=1.35,
            color=COLORS["ink"],
        )
    return patch


def arrow(ax, start, end, color=None, style="-|>", lw=1.35, linestyle="-"):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=12,
        linewidth=lw,
        linestyle=linestyle,
        color=color or COLORS["line"],
        shrinkA=3,
        shrinkB=3,
    )
    ax.add_patch(patch)
    return patch


fig, ax = plt.subplots(figsize=(14.6, 7.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8)
ax.axis("off")

# Panel A: observed inputs.
ax.text(0.25, 7.48, "a  Pre-event inputs", fontsize=11.4, fontweight="bold")
box(
    ax,
    0.25,
    5.25,
    2.55,
    1.42,
    COLORS["input"],
    COLORS["line"],
    "Product multimodal content",
    r"Image $V_i$" "\n" r"Text and reviews $W_i$",
)
box(
    ax,
    0.25,
    3.40,
    2.55,
    1.42,
    COLORS["input"],
    COLORS["line"],
    "Marketplace information",
    r"Social volume $P_{i,<t}$" "\n" r"Mean rating $R_{i,<t}$",
)
box(
    ax,
    0.25,
    1.55,
    2.55,
    1.42,
    COLORS["input"],
    COLORS["line"],
    "Consumer history",
    r"Prior purchases $B_{u,<t}$" "\n" r"Candidate taxonomy $c_i$",
)

# Panel B: MCRE inventory.
ax.text(3.25, 7.48, "b  Theory-guided MCRE cue inventory", fontsize=11.4, fontweight="bold")
box(
    ax,
    3.25,
    1.1,
    3.35,
    5.55,
    COLORS["mnc"],
    COLORS["mnc_dark"],
    "MNC pathway",
    "Promotional and social influence\n\n"
    r"$T_{\mathrm{con,mkt}}$  Marketing salience" "\n"
    r"  $a_{\mathrm{pri}}$ Price shock" "\n"
    r"  $a_{\mathrm{gft}}$ Gift appeal" "\n"
    r"  $a_{\mathrm{sub}}$ Subsidy authenticity" "\n"
    r"  $a_{\mathrm{urg}}$ Urgency" "\n\n"
    r"$T_{\mathrm{con,soc}}$  Social proof" "\n"
    r"$T_{\mathrm{con,rat}}$  Rating valence",
    title_color=COLORS["mnc_dark"],
    body_size=8.1,
)
box(
    ax,
    6.85,
    1.1,
    3.35,
    5.55,
    COLORS["mai"],
    COLORS["mai_dark"],
    "MAI pathway",
    "Issue-relevant product evaluation\n\n"
    r"$T_{\mathrm{int,vis}}$  Functional visuals" "\n"
    r"  $a_{\mathrm{spec}}$ Specification overlay" "\n"
    r"  $a_{\mathrm{str}}$ Internal-structure view" "\n\n"
    r"$T_{\mathrm{int,fac}}$  Factual density" "\n"
    r"$T_{\mathrm{int,sem}}$  Historical category affinity",
    title_color=COLORS["mai_dark"],
    body_size=8.1,
)
# Panel C: treatment specifications and DML.
ax.text(10.65, 7.48, "c  Estimation inputs", fontsize=11.4, fontweight="bold")
box(
    ax,
    10.65,
    5.05,
    4.0,
    1.38,
    COLORS["output"],
    COLORS["output_dark"],
    "Composite-level cue specification",
    r"$\mathbf{T}_{\mathrm{composite}}\in\mathbb{R}^{6}$" "\n"
    "Three MNC cues + three MAI cues",
    title_color=COLORS["output_dark"],
)
box(
    ax,
    10.65,
    3.30,
    4.0,
    1.38,
    COLORS["output"],
    COLORS["output_dark"],
    "Component-level cue specification",
    r"$\mathbf{T}_{\mathrm{component}}\in\mathbb{R}^{10}$" "\n"
    "Six components + four standalone cues",
    title_color=COLORS["output_dark"],
)
box(
    ax,
    11.05,
    1.25,
    3.2,
    1.38,
    COLORS["white"],
    COLORS["output_dark"],
    "Stage-specific DML",
    r"ATE and CATE for $Y_s$" "\n"
    r"adjust $\mathbf{X}_{<t}$" "\n"
    r"compare available funnel stages",
    title_color=COLORS["output_dark"],
)
# Flow arrows.
arrow(ax, (2.82, 4.10), (3.21, 4.10))
arrow(ax, (10.22, 4.75), (10.61, 4.75), color=COLORS["output_dark"])
ax.text(
    12.65,
    2.96,
    "Each specification estimated separately",
    ha="center",
    va="center",
    fontsize=8.0,
    color=COLORS["output_dark"],
)
arrow(ax, (12.65, 2.87), (12.65, 2.67), color=COLORS["output_dark"])

fig.tight_layout(pad=0.4)
pdf_path = OUT_DIR / "Figure_5_mcre_measurement_graph_20260723.pdf"
png_path = OUT_DIR / "Figure_5_mcre_measurement_graph_20260723.png"
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(png_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Wrote {pdf_path}")
print(f"Wrote {png_path}")
