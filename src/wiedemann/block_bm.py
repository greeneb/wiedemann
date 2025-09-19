"""
Block Berlekamp-Massey Algorithm Implementation

This module implements the block version of the Berlekamp-Massey algorithm
for finding recurrence relations in block sequences. This is used in the
block Wiedemann algorithm for solving large linear systems over finite fields.

The block Berlekamp-Massey algorithm finds coefficient matrices that define
a recurrence relation for a sequence of matrices, which is more efficient
than applying the scalar algorithm to each element.

References:
- Kaltofen, E. (1993). Analysis of Coppersmith's block Wiedemann algorithm.
- Coppersmith, D. (1994). Solving homogeneous linear equations over GF(2).
"""

import galois
import numpy as np

def block_berlekamp_massey(S, field):
    """
    Block Berlekamp–Massey (BBM) via block Hankel + nullspace over GF(p).

    Args:
        S (list[galois.FieldArray]): Sequence of m x m matrices S_t = U^T A^t V.
        field (galois.GF): Field definition.

    Returns:
        list[galois.FieldArray]: Coefficient matrices [Q_0, Q_1, ..., Q_L] defining the recurrence
                                 Q_0 S_t + Q_1 S_{t+1} + ... + Q_L S_{t+L} = 0.
    """
    m = S[0].shape[0]
    n_terms = len(S)

    # Choose L = m (block size heuristic)
    L = m
    if n_terms < 2 * L:
        raise ValueError("Need at least 2*L block terms for BBM")

    # Build block Hankel matrix H of size (L) x ((L+1) * m*m)
    # Each row corresponds to a flattened sequence of (L+1) blocks
    rows = []
    for i in range(L):
        row_blocks = []
        for j in range(L + 1):
            block = S[i + j].view(np.ndarray).reshape(1, -1)  # flatten m×m → length m*m
            row_blocks.append(block)
        row = np.hstack(row_blocks)
        rows.append(row)
    H = np.vstack(rows)
    H = field(H % field.characteristic)

    # Compute one nullspace vector
    nullvec = null_space_mod_p(H, field)
    if nullvec is None:
        raise ValueError("No nontrivial nullspace found; sequence may be too short")

    # Reshape into (L+1) blocks of size m x m
    coeffs = []
    block_size = m * m
    expected_len = (L + 1) * block_size
    if nullvec.size != expected_len:
        raise ValueError(f"Unexpected nullspace vector size {nullvec.size}, expected {expected_len}")

    for i in range(L + 1):
        block_flat = nullvec[i * block_size : (i + 1) * block_size]
        block = np.array(block_flat).reshape(m, m)
        coeffs.append(field(block))

    return coeffs

def null_space_mod_p(A, field):
    """
    Simple nullspace computation over GF(p) using Gaussian elimination.
    Returns one nontrivial nullspace vector if it exists, else None.
    """
    A = A.copy()
    n_rows, n_cols = A.shape
    pivots = []
    row = 0

    for col in range(n_cols):
        # Find pivot
        pivot = None
        for r in range(row, n_rows):
            if A[r, col] != 0:
                pivot = r
                break
        if pivot is None:
            continue
        # Swap rows
        if pivot != row:
            A[[row, pivot], :] = A[[pivot, row], :]
        # Normalize pivot row
        inv = field(1) / A[row, col]
        A[row, :] *= inv
        # Eliminate other rows
        for r in range(n_rows):
            if r != row and A[r, col] != 0:
                factor = A[r, col]
                A[r, :] -= factor * A[row, :]
        pivots.append(col)
        row += 1
        if row == n_rows:
            break

    free_cols = [c for c in range(n_cols) if c not in pivots]
    if not free_cols:
        return None

    # Construct one nullspace vector by setting one free variable = 1
    ns_vec = field.Zeros(n_cols)
    free = free_cols[0]
    ns_vec[free] = 1
    # Back substitution
    for i, col in enumerate(reversed(pivots)):
        r = row - 1 - i
        val = -sum((A[r, c] * ns_vec[c] for c in free_cols), field(0))
        ns_vec[col] = val
    return ns_vec