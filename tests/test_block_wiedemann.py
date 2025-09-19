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

def test_large_singular_matrix():
    GF = galois.GF(7)
    # Create a 6x6 singular matrix with rank 4
    A = GF([[1, 2, 3, 4, 5, 6],
            [2, 4, 6, 1, 3, 5],
            [3, 6, 2, 5, 1, 4],
            [4, 1, 5, 2, 6, 3],
            [1, 2, 3, 4, 5, 6],  # duplicate of first row
            [0, 0, 0, 0, 0, 0]])  # zero row
    
    w = block_wiedemann(A, GF, m=3, n=3, max_iter=15)
    assert w.shape == (6,)
    assert not np.all(w == 0)
    assert (A @ w == 0).all()

def test_different_block_sizes():
    GF = galois.GF(5)
    A = GF([[1, 2, 3], [2, 4, 1], [1, 1, 1]])  # Singular 3x3 matrix
    
    # Test with different block sizes
    for m, n in [(1, 1), (2, 2), (3, 3)]:
        w = block_wiedemann(A, GF, m=m, n=n, max_iter=10)
        assert w.shape == (3,)
        assert not np.all(w == 0)
        assert (A @ w == 0).all()

def test_zero_matrix():
    GF = galois.GF(3)
    A = GF([[0, 0, 0], [0, 0, 0], [0, 0, 0]])  # Zero matrix
    
    w = block_wiedemann(A, GF, m=2, n=2, max_iter=8)
    assert w.shape == (3,)
    assert not np.all(w == 0)
    assert (A @ w == 0).all()