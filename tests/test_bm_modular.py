import galois

from wiedemann.experimental import (
    berlekamp_massey as bm_experimental,
    minimal_polynomial_wiedemann,
    minimal_polynomial_wiedemann_blackbox,
)


def annihilates_sequence(poly, seq):
    GF = type(seq)
    coeffs = poly.coeffs
    degree = poly.degree
    for idx in range(degree, len(seq)):
        value = GF(0)
        for offset, coeff in enumerate(coeffs):
            value += coeff * seq[idx - degree + offset]
        if value != 0:
            return False
    return True


def test_berlekamp_massey_annihilates_sequence():
    GF = galois.GF(5)
    seq = GF.Random(20, seed=123)

    poly = bm_experimental(seq)

    assert annihilates_sequence(poly, seq)
    assert poly.degree >= 0


def test_blackbox_minpoly_matches_explicit():
    GF = galois.GF(4)
    A = GF([[1, 1, 0], [0, 1, 1], [1, 0, 1]])

    poly_matrix = minimal_polynomial_wiedemann(A, tries=4, seed=7)

    def matvec(vec):
        return A @ vec

    poly_blackbox = minimal_polynomial_wiedemann_blackbox(
        matvec, GF, A.shape[0], tries=4, seed=7
    )

    assert poly_blackbox == poly_matrix
