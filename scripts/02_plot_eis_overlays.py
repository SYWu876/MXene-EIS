from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.nanoenergy_style import apply_style, nanoenergy_axes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

def main():
    apply_style()
    df = pd.read_csv(DATA / "MXene_EIS_fit_results_linearLS.csv")
    f = df["Frequency_Hz"].to_numpy(float)
    Z = df["Zre_data_Ohm"].to_numpy(float) + 1j*df["Zim_data_Ohm"].to_numpy(float)
    Zfit = df["Zre_fit_Ohm"].to_numpy(float) + 1j*df["Zim_fit_Ohm"].to_numpy(float)
    order = np.argsort(f)
    f = f[order]; Z = Z[order]; Zfit = Zfit[order]

    fig, ax = plt.subplots(figsize=(3.6, 3.6))
    ax.plot(Z.real, -Z.imag, marker="o", markersize=3, linestyle="None", label="Data")
    ax.plot(Zfit.real, -Zfit.imag, linewidth=1.2, label="Fit")
    ax.set_xlabel("Z' (Ω)")
    ax.set_ylabel("−Z'' (Ω)")
    ax.set_aspect("equal", adjustable="box")
    nanoenergy_axes(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "Regen_Fig3_Nyquist_Overlay.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Regen_Fig3_Nyquist_Overlay.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.semilogx(f, np.abs(Z), marker="o", markersize=3, linestyle="None", label="Data")
    ax.semilogx(f, np.abs(Zfit), linewidth=1.2, label="Fit")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|Z| (Ω)")
    nanoenergy_axes(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "Regen_Fig3_BodeMag_Overlay.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Regen_Fig3_BodeMag_Overlay.pdf", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    ax.semilogx(f, np.angle(Z, deg=True), marker="o", markersize=3, linestyle="None", label="Data")
    ax.semilogx(f, np.angle(Zfit, deg=True), linewidth=1.2, label="Fit")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (deg)")
    nanoenergy_axes(ax)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "Regen_Fig3_BodePhase_Overlay.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Regen_Fig3_BodePhase_Overlay.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Saved regenerated overlays to:", FIG)

if __name__ == "__main__":
    main()
