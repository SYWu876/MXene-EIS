from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from src.mxene_eis_model import Z_model, residual_vec
from src.bounds_decoding import load_bounds_csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def main():
    df_raw = pd.read_csv(DATA / "MXene EIS.csv")
    f = df_raw.iloc[:, 0].to_numpy(float)
    zre = df_raw.iloc[:, 1].to_numpy(float)
    zim = df_raw.iloc[:, 2].to_numpy(float)

    order = np.argsort(f)
    f = f[order]; zre = zre[order]; zim = zim[order]
    Z = zre + 1j*zim
    w = 2*np.pi*f

    _, space, lb, ub = load_bounds_csv(DATA / "MXene_Qbranch_bounds_decoding_TableS1style.csv")

    theta0 = np.array([8.72511388e-01, 1.87992532e-07, 1.50028986e+01,
                       2.62172998e-03, 8.01697209e-01, 2.11738227e-03, 9.00106897e-01], float)
    theta0 = np.clip(theta0, lb, ub)

    res = least_squares(lambda th: residual_vec(th, w, Z),
                        x0=theta0, bounds=(lb, ub), method="trf",
                        max_nfev=2000, ftol=1e-10, xtol=1e-10, gtol=1e-10)
    theta = res.x
    Zfit = Z_model(theta, w)

    out = pd.DataFrame({
        "Frequency_Hz": f,
        "Zre_data_Ohm": Z.real,
        "Zim_data_Ohm": Z.imag,
        "Zre_fit_Ohm": Zfit.real,
        "Zim_fit_Ohm": Zfit.imag,
    })
    out.to_csv(DATA / "MXene_EIS_fit_results_linearLS.csv", index=False)
    print("Saved:", DATA / "MXene_EIS_fit_results_linearLS.csv")
    print("nfev:", res.nfev)

if __name__ == "__main__":
    main()
