from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.nanoenergy_style import apply_style, nanoenergy_axes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

def heatmap(df: pd.DataFrame, out_base: str, mark=None):
    g = np.sort(df["gamma"].unique())
    b = np.sort(df["beta"].unique())
    E = df.pivot(index="beta", columns="gamma", values="Expected_H").loc[b, g].to_numpy()
    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    im = ax.imshow(E, origin="lower", aspect="auto",
                   extent=[float(g.min()), float(g.max()), float(b.min()), float(b.max())],
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Expected ⟨H⟩")
    if mark is not None:
        ax.plot(mark[0], mark[1], marker="*", markersize=10, linestyle="None")
        ax.text(mark[0], mark[1], "  (γ*,β*)", ha="left", va="center", fontsize=9)
    ax.set_xlabel("γ")
    ax.set_ylabel("β")
    nanoenergy_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / f"{out_base}.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / f"{out_base}.pdf", bbox_inches="tight")
    plt.close(fig)

def main():
    apply_style()
    df_coarse = pd.read_csv(DATA / "MXene_QAOA_gammabeta_heatmap_d008.csv")
    df_ref = pd.read_csv(DATA / "MXene_QAOA_gammabeta_heatmap_refined9x9_d008.csv")
    best = df_ref.loc[df_ref["Expected_H"].idxmin()]
    mark = (float(best["gamma"]), float(best["beta"]))
    heatmap(df_coarse, "Regen_FigS8_QAOA_heatmap_coarse", mark=mark)
    heatmap(df_ref, "Regen_FigS8_QAOA_heatmap_refined9x9", mark=mark)
    print("Saved regenerated heatmaps to:", FIG)

if __name__ == "__main__":
    main()
