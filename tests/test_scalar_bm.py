import pytest
import galois
from wiedemann.scalar_bm import berlekamp_massey

def annihilates(poly, seq, GF):
    """Check if a polynomial annihilates the given sequence over GF."""
    coeffs = poly.coeffs
    d = poly.degree
    for k in range(d, len(seq)):
        val = GF(0)
        for j, c in enumerate(coeffs):
            val += c * seq[k - d + j]
        if val != 0:
            return False
    return True

def test_linear_sequence_gf2():
    GF = galois.GF(2)
    seq = [1, 0, 1, 0, 1, 0, 1, 0]  # Periodic sequence (period 2)
    poly = berlekamp_massey(seq, GF)
    assert poly.degree == 2
    assert annihilates(poly, seq, GF)

def test_repeated_sequence_gf5():
    GF = galois.GF(5)
    seq = [1, 1, 1, 1, 1, 1, 1, 1]  # Constant sequence
    poly = berlekamp_massey(seq, GF)
    assert poly.degree == 1
    assert annihilates(poly, seq, GF)

def test_fibonacci_like_sequence_gf5():
    GF = galois.GF(5)
    # Define a sequence by recurrence: a_n = a_{n-1} + a_{n-2}
    seq = [1, 1, 2, 3, 0, 3, 3, 1, 4, 0, 4, 4, 3]
    poly = berlekamp_massey(seq, GF)
    assert poly.degree == 2
    assert annihilates(poly, seq, GF)

def test_short_sequence_returns_valid_poly():
    GF = galois.GF(7)
    seq = [3, 5]
    poly = berlekamp_massey(seq, GF)
    assert isinstance(poly, galois.Poly)
    assert poly.field == GF
    assert annihilates(poly, seq, GF)