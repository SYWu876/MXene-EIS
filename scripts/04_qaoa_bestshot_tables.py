from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from src.bounds_decoding import load_bounds_csv, decode_u_to_theta, encode_theta_to_u
from src.qaoa_tools import ising_energy_all_states, zsum_all_states, qaoa_p1_probs, sample_from_prob
from src.mxene_eis_model import sse_complex
from src.nanoenergy_style import apply_style, nanoenergy_axes

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIG = ROOT / "figures"

def bitstring(state, m=21):
    return format(int(state), f"0{m}b")

def decode_state_to_theta(state, u0, Delta, lb, ub, space):
    bits = np.array([(state >> j) & 1 for j in range(21)], dtype=int)
    s = np.zeros(7, dtype=float)
    for i in range(7):
        b0, b1, b2 = bits[3*i:3*i+3]
        s[i] = (b0 + 2*b1 + 4*b2) / 7.0
    u = np.clip(u0 + Delta*(2*s - 1), 0.0, 1.0)
    return decode_u_to_theta(u, lb, ub, space)

def main():
    apply_style()

    h = pd.read_csv(DATA / "MXene_QAOA_Ising_h.csv").sort_values("i")["h_i"].to_numpy(np.float32)
    Jdf = pd.read_csv(DATA / "MXene_QAOA_Ising_J.csv")
    m = 21
    J = np.zeros((m, m), dtype=np.float32)
    for _, r in Jdf.iterrows():
        i, j = int(r["i"]), int(r["j"])
        Jij = float(r["J_ij"])
        J[i, j] = Jij
        J[j, i] = Jij

    energy = ising_energy_all_states(h, J)
    zsum = zsum_all_states(m)

    ref = pd.read_csv(DATA / "MXene_QAOA_gammabeta_heatmap_refined9x9_d008.csv")
    best = ref.loc[ref["Expected_H"].idxmin()]
    gamma, beta = float(best["gamma"]), float(best["beta"])

    prob = qaoa_p1_probs(energy, zsum, gamma, beta)
    samples, counts = sample_from_prob(prob, shots=4096, seed=7)

    best_state = min(counts.keys(), key=lambda s: float(energy[int(s)]))
    top10 = counts.most_common(10)

    names, space, lb, ub = load_bounds_csv(DATA / "MXene_Qbranch_bounds_decoding_TableS1style.csv")
    theta_seed = np.array([8.72511388e-01, 1.87992532e-07, 1.50028986e+01,
                           2.62172998e-03, 8.01697209e-01, 2.11738227e-03, 9.00106897e-01], float)
    u0 = encode_theta_to_u(theta_seed, lb, ub, space)
    Delta = 0.08
    theta_best = decode_state_to_theta(best_state, u0, Delta, lb, ub, space)

    # SSE on clean data in fit file
    df_fit = pd.read_csv(DATA / "MXene_EIS_fit_results_linearLS.csv")
    f = df_fit["Frequency_Hz"].to_numpy(float)
    Z = df_fit["Zre_data_Ohm"].to_numpy(float) + 1j*df_fit["Zim_data_Ohm"].to_numpy(float)
    order = np.argsort(f)
    f = f[order]; Z = Z[order]
    w = 2*np.pi*f
    sse = sse_complex(theta_best, w, Z)

    pd.DataFrame({"Parameter": names, "Value": theta_best}).to_csv(
        DATA / "Regen_FigureS9_bestshot_theta_table.csv", index=False
    )

    top10_df = pd.DataFrame({
        "rank": np.arange(1, len(top10) + 1),
        "state_index": [int(s) for s, _ in top10],
        "bitstring_q20_to_q0": [bitstring(s, m) for s, _ in top10],
        "counts": [int(c) for _, c in top10],
        "ising_energy": [float(energy[int(s)]) for s, _ in top10],
    })
    top10_df.to_csv(DATA / "Regen_FigureS9_top10_bitstrings_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(top10_df))
    ax.bar(x, top10_df["counts"].to_numpy(int))
    ax.set_xlabel("Bitstring (q20…q0)")
    ax.set_ylabel("Counts (shots=4096)")
    ax.set_xticks(x)
    ax.set_xticklabels(top10_df["bitstring_q20_to_q0"].tolist(), rotation=90, ha="center", fontsize=7)
    if best_state in top10_df["state_index"].to_numpy(int):
        idx = int(np.where(top10_df["state_index"].to_numpy(int) == best_state)[0][0])
        ax.plot(idx, top10_df["counts"].iloc[idx], marker="*", markersize=10, linestyle="None")
    nanoenergy_axes(ax)
    fig.tight_layout()
    fig.savefig(FIG / "Regen_FigS9_Top10BitstringCounts.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIG / "Regen_FigS9_Top10BitstringCounts.pdf", bbox_inches="tight")
    plt.close(fig)

    print("Best-shot state:", best_state, "bitstring:", bitstring(best_state, m))
    print("SSE(clean):", sse)

if __name__ == "__main__":
    main()
