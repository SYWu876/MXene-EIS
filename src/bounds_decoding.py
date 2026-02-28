"""Bounds + bounded decoding utilities used by all solvers."""

from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path

def load_bounds_csv(path: str | Path):
    df = pd.read_csv(path)
    names = df["Parameter"].tolist()
    space = df["Space"].tolist()
    lb = df["Lower bound (lb)"].to_numpy(float)
    ub = df["Upper bound (ub)"].to_numpy(float)
    return names, space, lb, ub

def decode_u_to_theta(u, lb, ub, space):
    """Map u∈[0,1]^d → theta using linear/log decoding per parameter."""
    u = np.clip(np.asarray(u, float), 0.0, 1.0)
    theta = np.zeros_like(u, dtype=float)
    for i, sp in enumerate(space):
        if sp == "log":
            theta[i] = 10 ** (np.log10(lb[i]) + u[i] * (np.log10(ub[i]) - np.log10(lb[i])))
        else:
            theta[i] = lb[i] + u[i] * (ub[i] - lb[i])
    return theta

def encode_theta_to_u(theta, lb, ub, space):
    """Inverse map theta→u (clipped)."""
    theta = np.asarray(theta, float)
    u = np.zeros_like(theta, dtype=float)
    for i, sp in enumerate(space):
        if sp == "log":
            u[i] = (np.log10(theta[i]) - np.log10(lb[i])) / (np.log10(ub[i]) - np.log10(lb[i]))
        else:
            u[i] = (theta[i] - lb[i]) / (ub[i] - lb[i])
    return np.clip(u, 0.0, 1.0)
