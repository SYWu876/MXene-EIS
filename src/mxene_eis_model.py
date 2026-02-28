"""MXene EIS equivalent-circuit model and fit metrics.

Circuit: Z(w) = Rs + j*w*L + (Rct || CPE1) + CPE2
theta = [Rs, L, Rct, Q1, a1, Q2, a2]
"""

from __future__ import annotations
import numpy as np

def Z_cpe(Q: float, alpha: float, w: np.ndarray) -> np.ndarray:
    """Constant phase element: Z = 1 / (Q * (j*w)^alpha)."""
    return 1.0 / (Q * (1j * w) ** alpha)

def Z_model(theta: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Compute circuit impedance for the 7-parameter model."""
    Rs, L, Rct, Q1, a1, Q2, a2 = [float(x) for x in theta]
    ZL = 1j * w * L
    Z1 = Z_cpe(Q1, a1, w)
    Zpar = 1.0 / (1.0 / Rct + 1.0 / Z1)
    Z2 = Z_cpe(Q2, a2, w)
    return Rs + ZL + Zpar + Z2

def residual_vec(theta: np.ndarray, w: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """Stacked real+imag residuals for least squares."""
    r = Z_model(theta, w) - Z
    return np.r_[r.real, r.imag]

def sse_complex(theta: np.ndarray, w: np.ndarray, Z: np.ndarray) -> float:
    """Complex-domain SSE: sum |Zfit - Zdata|^2."""
    r = Z_model(theta, w) - Z
    return float(np.sum(r.real**2 + r.imag**2))

def rmse_complex(theta: np.ndarray, w: np.ndarray, Z: np.ndarray) -> float:
    return float(np.sqrt(sse_complex(theta, w, Z) / Z.size))

def r2_complex(theta: np.ndarray, w: np.ndarray, Z: np.ndarray) -> float:
    """R^2 computed on concatenated Re/Im components."""
    Zfit = Z_model(theta, w)
    y = np.r_[Z.real, Z.imag]
    yhat = np.r_[Zfit.real, Zfit.imag]
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
