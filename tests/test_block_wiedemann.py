import galois
import numpy as np
from wiedemann.block_wiedemann import block_krylov, block_wiedemann

def test_block_krylov_shapes():
    GF = galois.GF(5)
    A = GF([[1, 2, 0], [0, 1, 3], [4, 0, 2]])
    U = GF.Random((3, 2))
    V = GF.Random((3, 2))

    S = block_krylov(A, U, V, GF, max_iter=5)

    assert isinstance(S, list)
    assert len(S) == 5
    for M in S:
        assert M.shape == (2, 2)
        assert isinstance(M, GF)

def test_block_wiedemann_shapes():
    GF = galois.GF(7)
    A = GF([[1, 0, 0], [0, 2, 0], [0, 0, 3]])

    S, U, V = block_wiedemann(A, GF, m=2, max_iter=6)

    assert len(S) == 6
    for M in S:
        assert M.shape == (2, 2)
        assert isinstance(M, GF)
    assert U.shape == (3, 2)
    assert V.shape == (3, 2)
    assert isinstance(U, GF)
    assert isinstance(V, GF)

def test_block_krylov_recurrence_toy_case():
    GF = galois.GF(3)
    A = GF([[1, 0], [0, 0]])  # Rank-deficient diagonal
    U = GF([[1, 0], [0, 1]])
    V = GF([[1, 1], [1, 1]])

    S = block_krylov(A, U, V, GF, max_iter=4)

    # Because A^2 = A for this matrix, sequence should stabilize
    assert (S[0] == S[1]).all() or (S[1] == S[2]).all()