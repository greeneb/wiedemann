import galois
import numpy as np

def block_krylov(A, U, V, field, max_iter=None):
    """
    Construct the block Krylov sequence S_t = U^T A^t V.

    Args:
        A (galois.FieldArray): n x n matrix over GF(p).
        U (galois.FieldArray): n x m random block vector.
        V (galois.FieldArray): n x m random block vector.
        field (galois.GF): Field definition.
        max_iter (int, optional): Number of iterations. Default: 2*n.

    Returns:
        list[galois.FieldArray]: List of m x m matrices (the S_t sequence).
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = 2 * n

    W = V.copy()
    S = []
    for _ in range(max_iter):
        S.append(U.T @ W)
        W = A @ W

    return S

def block_wiedemann(A, field, m=2, max_iter=None):
    """
    Block Wiedemann setup: generate block Krylov sequence with random U, V.

    Args:
        A (galois.FieldArray): n x n matrix over GF(p).
        field (galois.GF): Field definition.
        m (int): Block size.
        max_iter (int, optional): Number of iterations. Default: 2*n.

    Returns:
        (list[galois.FieldArray], galois.FieldArray, galois.FieldArray):
            The sequence [S_0, S_1, ...], along with U, V.
    """
    n = A.shape[0]
    if max_iter is None:
        max_iter = 2 * n

    # Random block vectors
    U = field.Random((n, m))
    V = field.Random((n, m))

    S = block_krylov(A, U, V, field, max_iter=max_iter)
    return S, U, V