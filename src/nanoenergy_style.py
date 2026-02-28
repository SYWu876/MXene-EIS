"""Minimal Nano Energy-style matplotlib formatting."""

from __future__ import annotations
import matplotlib.pyplot as plt

def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.linewidth": 1.0,
        "lines.linewidth": 1.2,
        "savefig.dpi": 600,
    })

def nanoenergy_axes(ax):
    ax.tick_params(direction="in", which="both", top=True, right=True, length=4, width=1.0)
    ax.tick_params(which="minor", length=2)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)
    ax.minorticks_on()
