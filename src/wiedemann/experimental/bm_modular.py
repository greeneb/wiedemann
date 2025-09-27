#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# pip install galois
from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import galois

from wiedemann.scalar_bm import berlekamp_massey as _scalar_berlekamp

__all__ = [
    "make_gf4",
    "make_rng",
    "berlekamp_massey",
    "wiedemann_sequence",
    "minimal_polynomial_wiedemann",
    "minimal_polynomial_wiedemann_blackbox",
]

# =========================
# Field & RNG utilities
# =========================

def make_gf4() -> galois.FieldClass:
    """Return GF(4) (conway construction)."""
    return galois.GF(4)

def make_rng(seed: Optional[int] = None) -> np.random.Generator:
    """Deterministic RNG for reproducibility (optional)."""
    return np.random.default_rng(seed)


# =========================
# Berlekamp–Massey (sequence -> polynomial)
# =========================

def berlekamp_massey(sequence: galois.FieldArray) -> galois.Poly:
    """Delegate to the proven scalar Berlekamp–Massey implementation."""

    field = type(sequence)
    return _scalar_berlekamp(sequence, field)

# =========================
# Wiedemann core (matrix -> sequence)
# =========================

def wiedemann_sequence(
    matvec: Callable[[galois.FieldArray], galois.FieldArray],
    left: galois.FieldArray,
    right: galois.FieldArray,
    length: int,
) -> galois.FieldArray:
    """
    Build s_t = <left, A^t right> for t = 0..length-1 using only a matvec() oracle.
    Shapes:
        left  : (n,)
        right : (n,)
    """
    F = type(left)
    seq = F.Zeros(length)
    v = right.copy()
    for t in range(length):
        seq[t] = left @ v
        v = matvec(v)
    return seq


# =========================
# Minimal polynomial (Wiedemann + BM)
# =========================

def minimal_polynomial_wiedemann(
    A: galois.FieldArray,
    *,
    tries: int = 2,
    seed: Optional[int] = None,
) -> galois.Poly:
    """
    Minimal polynomial of square GF(4) matrix A via Wiedemann + BM.
    - Uses only matvecs: good constants, easy to black-box.
    - Repeats with random (left,right) and LCMs candidates.
    """
    assert A.shape[0] == A.shape[1], "A must be square"
    F = type(A)
    n = A.shape[0]
    rng = make_rng(seed)
    minpoly = galois.Poly([1], field=F)

    # matvec oracle
    def matvec(x: galois.FieldArray) -> galois.FieldArray:
        return A @ x

    for _ in range(max(1, tries)):
        left  = F.Random(n, seed=int(rng.integers(0, 2**32)))
        right = F.Random(n, seed=int(rng.integers(0, 2**32)))
        if np.all(left == 0) or np.all(right == 0):
            continue
        seq = wiedemann_sequence(matvec, left, right, length=2*n)
        cand = berlekamp_massey(seq)
        minpoly = galois.lcm(minpoly, cand)
        if minpoly.degree == n:
            break
    return minpoly


# =========================
# Optional: black-box API
# =========================

def minimal_polynomial_wiedemann_blackbox(
    matvec: Callable[[galois.FieldArray], galois.FieldArray],
    field: galois.FieldClass,
    n: int,
    *,
    tries: int = 2,
    seed: Optional[int] = None,
) -> galois.Poly:
    """
    Same as above, but takes a matvec oracle directly (no explicit matrix).
    Useful if A is implicit (e.g., projector, convolution operator).
    """
    rng = make_rng(seed)
    minpoly = galois.Poly([1], field=field)
    for _ in range(max(1, tries)):
        left  = field.Random(n, seed=int(rng.integers(0, 2**32)))
        right = field.Random(n, seed=int(rng.integers(0, 2**32)))
        if np.all(left == 0) or np.all(right == 0):
            continue
        seq = wiedemann_sequence(matvec, left, right, length=2*n)
        cand = berlekamp_massey(seq)
        minpoly = galois.lcm(minpoly, cand)
        if minpoly.degree == n:
            break
    return minpoly


# =========================
# Tiny self-check
# =========================

# if __name__ == "__main__":
#     GF4 = make_gf4()
#     A = GF4([[1,1,0,0],
#              [0,1,1,0],
#              [0,0,1,1],
#              [1,0,0,1]])

#     mp = minimal_polynomial_wiedemann(A, tries=3, seed=123)
#     print("minpoly(A):", mp)

#     # Verify p(A) annihilates a random vector
#     v = GF4.Random(A.shape[0])
#     acc = GF4.Zeros_like(v)
#     Av = v.copy()
#     for i, c in enumerate(mp.coeffs[::-1]):  # x^d + ... + c0
#         if i == mp.degree:  # x^d term
#             acc += Av
#         else:
#             acc += c * Av
#         Av = A @ Av
#     print("check p(A)·v == 0:", np.all(acc == 0))


def _demo() -> None:
    """Tiny manual check when running this module directly."""
    GF4 = make_gf4()
    A = GF4(
        [[1, 1, 0],
         [0, 1, 1],
         [1, 0, 1]]
    )

    mp = minimal_polynomial_wiedemann(A, tries=2, seed=42)
    print("Minimal polynomial of A:", mp)

    v = GF4.Random(A.shape[0], seed=42)
    acc = GF4.Zeros_like(v)
    Av = v.copy()
    for i, c in enumerate(mp.coeffs[::-1]):  # x^d + ... + c0
        if i == mp.degree:
            acc += Av
        else:
            acc += c * Av
        Av = A @ Av
    print("\nVerification that mp(A)·v = 0:", np.all(acc == 0))


if __name__ == "__main__":  # pragma: no cover - manual smoke test.
    _demo()
