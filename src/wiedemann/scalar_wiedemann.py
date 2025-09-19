"""
Scalar Wiedemann Algorithm Implementation

This module implements the scalar Wiedemann algorithm for finding kernel vectors
of matrices over finite fields. The algorithm is particularly useful for solving
large sparse linear systems and has applications in quantum error correction.

The scalar Wiedemann algorithm works by:
1. Generating a scalar sequence from the matrix using random projections
2. Finding the minimal polynomial of this sequence using Berlekamp-Massey
3. Using the minimal polynomial to construct a kernel vector

References:
- Wiedemann, D. (1986). Solving sparse linear equations over finite fields.
- Kaltofen, E. (1993). Analysis of Coppersmith's block Wiedemann algorithm.
"""

import galois
import numpy as np
from wiedemann.scalar_bm import berlekamp_massey

def scalar_wiedemann(A, field, x_base=None, max_iter=None):
    """
    Scalar Wiedemann algorithm for finding a minimal polynomial of the projected Krylov sequence.

    Args:
        A (galois.FieldArray): n x n matrix over GF(p).
        x_base (galois.FieldArray): n-dimensional starting vector. If None, a random one is chosen.
        field (galois.GF): Field definition.
        max_iter (int, optional): Maximum iterations for sequence length. Default: 2*n.

    Returns:
        galois.Poly: Minimal polynomial of the scalar sequence.
    """
    n = A.shape[0]
    if x_base is None:
        x_base = field.Random(n)
    if max_iter is None:
        max_iter = 2 * n

    # Compute x = A x_base
    x = A @ x_base

    # Pick random projection vector y
    y = field.Random(n)

    # Build scalar sequence s_t = y^T A^t x
    v = x.copy()
    sequence = []
    for _ in range(max_iter):
        s = int(y @ v)
        sequence.append(field(s))
        v = A @ v

    # Find minimal polynomial of sequence using BM
    poly = berlekamp_massey(sequence, field)

    return poly


def solve(A, field, x_base=None, max_attempts=10):
    """
    Find a kernel vector of A using scalar Wiedemann.

    Tries multiple random x_base until a non-zero kernel vector is found.

    Args:
        A (galois.FieldArray): n x n matrix.
        field (galois.GF): Field.
        x_base (galois.FieldArray, optional): Starting vector. If None, random vectors are chosen.
        max_attempts (int): Number of attempts with different x_base before failing.

    Returns:
        galois.FieldArray: A non-zero kernel vector v such that A v = 0.
    """
    n = A.shape[0]

    for attempt in range(max_attempts):
        if x_base is None:
            x_base_try = field.Random(n)
        else:
            x_base_try = x_base

        poly = scalar_wiedemann(A, field, x_base=x_base_try)
        coeffs = poly.coeffs
        d = poly.degree

        # If the polynomial is constant (degree 0), try a different approach
        if d == 0:
            # Try to find a kernel vector directly by testing random vectors
            for _ in range(5):
                v = field.Random(n)
                if not (v == 0).all() and (A @ v == 0).all():
                    return v
            continue

        # Build Krylov vectors of x_base_try
        krylov = [x_base_try]
        for _ in range(1, d + 1):
            krylov.append(A @ krylov[-1])

        # Form v = sum q_i M^i x_base_try
        v = field.Zeros(n)
        for j in range(len(coeffs)):
            if j < len(krylov):
                v += coeffs[j] * krylov[j]

        # Check if we found a valid kernel vector
        if not (v == 0).all() and (A @ v == 0).all():
            return v

    # If all attempts failed, try a more direct approach
    # Use Gaussian elimination to find the nullspace
    try:
        # Convert to numpy for nullspace computation
        A_np = A.view(np.ndarray)
        U, s, Vh = np.linalg.svd(A_np)
        # Find the nullspace vectors (columns of Vh corresponding to zero singular values)
        nullspace_indices = np.where(np.abs(s) < 1e-10)[0]
        if len(nullspace_indices) > 0:
            # Take the first nullspace vector
            nullspace_vec = Vh[nullspace_indices[0], :]
            # Convert back to field
            v = field(nullspace_vec)
            if not (v == 0).all() and (A @ v == 0).all():
                return v
    except:
        pass

    raise ValueError("Failed to produce a nonzero kernel vector after several attempts.")