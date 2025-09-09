import galois
import numpy as np
import pytest
from wiedemann.block_wiedemann import block_wiedemann


def test_small_singular_matrix_kernel():
    GF = galois.GF(5)
    A = GF([[1, 2], [2, 4]])  # rank 1, nullspace dimension 1

    w = block_wiedemann(A, GF, m=2, n=2, max_iter=6)
    assert w.shape == (2,)
    assert not np.all(w == 0)
    assert (A @ w == 0).all()

def test_3x3_singular_matrix_kernel():
    GF = galois.GF(7)
    A = GF([[1, 2, 3],
            [2, 4, 6],
            [1, 1, 1]])

    w = block_wiedemann(A, GF, m=2, n=2, max_iter=10)
    assert w.shape == (3,)
    assert not np.all(w == 0)
    assert (A @ w == 0).all()

def test_random_singular_matrix():
    GF = galois.GF(3)
    # Construct a 4x4 singular matrix (row repetition)
    A = GF([[1,0,0,0],
            [0,1,0,0],
            [1,0,0,0],  # duplicate of row 0
            [0,0,0,0]])

    w = block_wiedemann(A, GF, m=2, n=2, max_iter=12)
    assert w.shape == (4,)
    assert not np.all(w == 0)
    assert (A @ w == 0).all()

@pytest.mark.xfail(reason="Block Wiedemann not yet generalized to nonsingular Ax=b case")
def test_inhomogeneous_not_supported():
    GF = galois.GF(5)
    A = GF([[1, 1], [0, 2]])
    b = GF([2, 4])

    # Currently block_wiedemann only supports homogeneous systems.
    _ = block_wiedemann(A, GF, m=1, n=1, max_iter=6)