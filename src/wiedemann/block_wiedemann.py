import galois
import numpy as np
from wiedemann.block_bm import null_space_mod_p

def generate_sequence(A, m, n, field, length):
    """
    Step C1: Generate block Krylov sequence a^(i) = X^T A^i Y.

    Args:
        A (galois.FieldArray): N x N matrix.
        m (int): number of left projection vectors.
        n (int): number of right projection vectors.
        field (galois.GF): finite field.
        length (int): number of sequence elements to generate.

    Returns:
        list[galois.FieldArray]: sequence of m x n matrices.
        X (galois.FieldArray): N x m random matrix.
        Y (galois.FieldArray): N x n random matrix.
    """
    N = A.shape[0]

    X = field.Random((N, m))
    Y = field.Random((N, n))

    S = []
    W = Y.copy()
    for _ in range(length):
        S.append(X.T @ W)
        W = A @ W
    return S, X, Y


def solve_block_toeplitz(S, field, m, n, N):
    """
    Step C2: Solve block Toeplitz linear system to find recurrence coefficients.

    Args:
        S (list[galois.FieldArray]): sequence of m x n matrices.
        field (galois.GF): finite field.
        m (int): block size (left).
        n (int): block size (right).
        N (int): matrix dimension.

    Returns:
        list[galois.FieldArray]: coefficient blocks c^(i).
    """
    # Choose parameters according to Kaltofen (1993)
    D = int(np.ceil(N / n))
    S_len = D + 1
    E = int(np.ceil(S_len / m))
    R = m * E

    # Build block Toeplitz system of size (R x S_len*n)
    rows = []
    for i in range(E):
        row_blocks = []
        for j in range(S_len):
            row_blocks.append(S[i + j].view(np.ndarray).reshape(-1))
        row = np.concatenate(row_blocks)
        rows.append(row)
    T = np.vstack(rows)
    T = field(T % field.characteristic)

    # Solve nullspace
    nullvec = null_space_mod_p(T, field)
    if nullvec is None:
        return None

    # Partition into coefficient blocks (each length n)
    coeffs = []
    for i in range(0, len(nullvec), n):
        coeffs.append(field(np.array(nullvec[i:i+n])))
    return coeffs


def reconstruct_solution(A, Y, coeffs, field):
    """
    Step C3: Reconstruct kernel vector from recurrence coefficients.
    """
    N, n = Y.shape
    wb = field.Zeros(N)
    for i, c in enumerate(coeffs):
        if not np.all(c == 0):
            vec = (A**i) @ (Y @ c.reshape(-1, 1))
            wb += vec.reshape(-1)

    if np.all(wb == 0):
        return wb

    # Backtrack until we hit the kernel
    v = wb
    for _ in range(len(coeffs) + 1):
        Av = A @ v
        if np.all(Av == 0):
            return v
        v = Av
    return None


def block_wiedemann(A, field, m=2, n=2, max_iter=None, max_retries=5):
    """
    Full Block Wiedemann algorithm (Kaltofen 1993, Steps C1–C3).

    Args:
        A (galois.FieldArray): N x N matrix.
        field (galois.GF): finite field.
        m (int): number of left projection vectors.
        n (int): number of right projection vectors.
        max_iter (int, optional): length of sequence to generate.
        max_retries (int): maximum retries with new random projections.

    Returns:
        galois.FieldArray: candidate kernel vector w.
    """
    N = A.shape[0]

    for attempt in range(max_retries):
        # Step C1: generate sequence length ~ N/m + N/n
        length = int(np.ceil(N/m) + np.ceil(N/n) + 2)
        if max_iter is not None:
            length = max(length, max_iter)
        S, X, Y = generate_sequence(A, m, n, field, length)

        # Step C2: solve block Toeplitz
        coeffs = solve_block_toeplitz(S, field, m, n, N)
        if coeffs is None:
            continue

        # Step C3: reconstruct solution
        w = reconstruct_solution(A, Y, coeffs, field)
        if w is not None and not np.all(w == 0):
            if (A @ w == 0).all():
                return w

    raise ValueError("Block Wiedemann failed after retries.")