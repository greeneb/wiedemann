import pytest
import galois
import numpy as np
from wiedemann.scalar_wiedemann import solve, scalar_wiedemann

# This one seems to fail a lot more often, 
# possibly due to the structure of the kernel? 
# Hence why Block Wiedemann exists.
def test_kernel_vector_exists_small_matrix():
    GF = galois.GF(5)
    # Rank-deficient 2x2 matrix (second row = multiple of first)
    A = GF([[1, 2],
            [2, 4]])
    v = solve(A, GF)
    assert v.shape == (2,)
    assert not np.all(v == 0)
    assert (A @ v == 0).all()

def test_kernel_vector_exists_3x3():
    GF = galois.GF(7)
    # Rank-deficient 3x3 matrix
    A = GF([[1, 2, 3],
            [2, 4, 6],
            [1, 1, 1]])
    v = solve(A, GF)
    assert v.shape == (3,)
    assert not np.all(v == 0)
    assert (A @ v == 0).all()

def test_random_singular_matrix_kernel():
    GF = galois.GF(3)
    # Construct a singular matrix with dependent rows
    A = GF([[1, 0, 2],
            [2, 0, 1],
            [1, 0, 2]])  # Row 3 == Row 1
    v = solve(A, GF)
    assert v.shape == (3,)
    assert not np.all(v == 0)
    assert (A @ v == 0).all()

def test_invertible_matrix_raises():
    GF = galois.GF(5)
    A = GF([[1, 2], [3, 4]])  # Invertible
    try:
        v = solve(A, GF)
    except ValueError:
        # Expected: no kernel vector should be found
        assert True
    else:
        # If we got here, verify A v != 0 (since kernel is trivial)
        assert not (A @ v == 0).all()