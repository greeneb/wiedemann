import numpy as np
import copy

class bm:
    """
    Implements the Berlekamp-Massey algorithm to find the shortest linear feedback shift register (LFSR) for a given sequence.
    The algorithm is used to find the minimal polynomial that generates a given binary sequence over GF(2).
    The minimal polynomial is the polynomial of the smallest degree that has the sequence as its coefficients.
    """

    @staticmethod
    def min_poly(sequence):
        """
            Find the minimal polynomial of a binary sequence using the Berlekamp-Massey algorithm.

            Args:
                sequence: A binary sequence (list of 0s and 1s)

            Returns:
                The coefficients of the minimal polynomial as a list
        """
        
        # Standard Berlekamp-Massey algorithm over GF(2)
        N = len(sequence)
        s = sequence
        C = [1]         # connection polynomial
        B = [1]         # last "best" copy of C
        L = 0           # current LFSR length
        m = 1           # steps since last update of B

        for n in range(N):
            # compute discrepancy d = s[n] + sum_{i=1..L} C[i]*s[n-i]
            d = s[n]
            for i in range(1, L + 1):
                if i < len(C) and C[i]:
                    d ^= s[n - i]

            if d:  # non-zero discrepancy, adjust C
                T = C.copy()  # save current C

                # C = C + x^m * B
                shift = m
                if len(C) < len(B) + shift:
                    C.extend([0] * (len(B) + shift - len(C)))
                for j, coeff in enumerate(B):
                    C[j + shift] ^= coeff

                # update L, B, and reset m if needed
                if 2 * L <= n:
                    B = T
                    L = n + 1 - L
                    m = 1
                else:
                    m += 1
            else:
                m += 1

        return C