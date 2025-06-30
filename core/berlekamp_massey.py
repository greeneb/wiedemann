import numpy as np
import copy
import galois


class BerlekampMassey:
    def __init__(self, field):
        """
        Initialize with a galois.GF field (e.g., GF = galois.GF(2))
        """
        self.field = field
        self.zero = field(0)
        self.one = field(1)

    def run(self, sequence, return_poly=False):
        """
        Run Berlekamp–Massey on a given sequence (list or FieldArray).

        Parameters:
            sequence : list or FieldArray of field elements
            return_poly : if True, returns a galois.Poly object

        Returns:
            - If return_poly: a galois.Poly minimal polynomial f(x)
            - Else: list of coefficients [c_0, c_1, ..., c_L]
        """
        n = len(sequence)
        c = [self.zero] * n
        b = [self.zero] * n
        c[0] = b[0] = self.one
        l = 0
        m = -1
        delta = self.one

        for i in range(n):
            # Compute discrepancy
            d = sequence[i]
            for j in range(1, l + 1):
                d += c[j] * sequence[i - j]  # galois handles modulo internally

            if d != self.zero:
                t = c.copy()
                scale = d / delta
                for j in range(i - m, n):
                    if j - (i - m) < len(b):
                        c[j] += scale * b[j - (i - m)]
                if 2 * l <= i:
                    l = i + 1 - l
                    m = i
                    b = t
                    delta = d

        coeffs = self.field(c[: l + 1])
        if return_poly:
            return galois.Poly(
                coeffs[::-1], field=self.field
            )  # descending order for x^d + ...
        else:
            return coeffs
