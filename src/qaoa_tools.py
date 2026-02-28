"""QAOA utilities: Ising energy, p=1 simulator via FWHT, and sampling."""

from __future__ import annotations
import numpy as np
from collections import Counter

def fwht_inplace(a: np.ndarray) -> np.ndarray:
    """Fast Walsh–Hadamard transform (in-place). Length must be power of 2."""
    h = 1
    n = a.shape[0]
    while h < n:
        a = a.reshape(-1, 2*h)
        x = a[:, :h].copy()
        y = a[:, h:]
        a[:, :h] = x + y
        a[:, h:] = x - y
        a = a.reshape(n)
        h *= 2
    return a

def ising_energy_all_states(h: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Compute Ising energy E(z) for all 2^m states."""
    m = len(h)
    N = 1 << m
    idx = np.arange(N, dtype=np.uint32)
    shifts = np.arange(m, dtype=np.uint32)
    bits = ((idx[:, None] >> shifts) & 1).astype(np.float32)
    z = 1.0 - 2.0 * bits
    e = z @ h
    e += 0.5 * np.sum(z * (z @ J), axis=1)
    return e.astype(np.float32)

def zsum_all_states(m: int) -> np.ndarray:
    """Return Σ Z_i for each computational basis state."""
    N = 1 << m
    idx = np.arange(N, dtype=np.uint32)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    bytes_view = idx.view(np.uint8).reshape(-1, 4)
    pop = lut[bytes_view].sum(axis=1).astype(np.int16)
    return (m - 2 * pop).astype(np.float32)

def qaoa_p1_probs(energy: np.ndarray, zsum: np.ndarray, gamma: float, beta: float) -> np.ndarray:
    """Probability distribution over states for p=1 QAOA."""
    N = energy.size
    sqrtN = np.sqrt(np.float32(N)).astype(np.float32)
    psi = (np.ones(N, dtype=np.complex64) / sqrtN)
    psi *= np.exp(-1j*np.float32(gamma) * energy).astype(np.complex64)
    psi = fwht_inplace(psi) / sqrtN
    psi *= np.exp(-1j*np.float32(beta) * zsum).astype(np.complex64)
    psi = fwht_inplace(psi) / sqrtN
    prob = (psi.real.astype(np.float32)**2 + psi.imag.astype(np.float32)**2).astype(np.float64)
    prob /= prob.sum()
    return prob

def sample_from_prob(prob: np.ndarray, shots: int = 4096, seed: int = 7):
    cdf = np.cumsum(prob); cdf[-1] = 1.0
    rng = np.random.default_rng(seed)
    samples = np.searchsorted(cdf, rng.random(shots), side="right").astype(np.int64)
    return samples, Counter(samples.tolist())
