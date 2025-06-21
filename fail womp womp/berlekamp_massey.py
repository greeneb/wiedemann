import copy
import numpy as np

class berlekamp_massey:
    """
    Implementation of the Berlekamp-Massey algorithm for finding the minimal polynomial
    of a sequence over GF(2).
    """
    
    @staticmethod
    def find_minimal_polynomial(sequence):
        """
        Apply the Berlekamp-Massey algorithm to find the minimal polynomial of a sequence.
        https://en.wikipedia.org/wiki/Berlekamp%E2%80%93Massey_algorithm
        
        Args:
            sequence: A list of binary values (0s and 1s)
            
        Returns:
            A list representing the minimal polynomial (connection polynomial)
        """
        N = len(sequence)               # length of the input sequence
        s = list(map(int, sequence))    # coeffs are s_j; output sequence as s_0 + s_1*x + s_2*x^2 + ..., N-1 degree polynomial
        C = np.zeros(N)                 # coeffs are c_j
        B = np.zeros(N)                 # Initialize polynomials to have right length
        C[0], B[0] = 1, 1               # Initial connection polynomials C(x), B(x) = 1
        L = 0                           # Length of current LFSR
        m = -1                          # Position of last discrepancy
        b = 1                           # Initial discrepancy
        
        for n in range(N):
            v = s[(n-L):n]
            v = v[::-1]  # Reverse the last L elements
            
            CC = C[1:L + 1]  # Current connection polynomial coefficients (excluding the first)
            
            d = s[n] + np.dot(v, CC) % 2 # Calculate discrepancy
            
            if d:  # If there is a discrepancy
                T = copy.copy(C)  # Save current connection polynomial
                
                p = np.zeros(N)  # Create a zero array for the update polynomial
                for j in range(L):
                    if B[j]:
                        p[j + n - m] = 1
                        
                # Update connection polynomial C(x) = C(x) + p(x)
                C = (C + p) % 2 # In GF(2), addition is XOR
                
                if 2 * L <= n:  # If current LFSR is not long enough
                    L = n + 1 - L
                    m = N
                    B = copy.copy(T)  # Update B with the last best polynomial
                    
        return C[:L + 1].tolist()  # Return the minimal polynomial coefficients up to degree L
            
        #     # If odd step number, discrepancy == 0, no need to calculate it. This is an artifact of GF(2) arithmetic
        #     # if n % 2 == 1:
        #     #     m += 1
        #     #     continue
            
        #     # Calculate discrepancy, we need to update the connection polynomial
        #     d = s[n]
        #     for i in range(1, L + 1):
        #         if i < len(C) and C[i]:#  == 1: and n - i >= 0:
        #             d ^= s[n - i]
            
        #     if d: # If there is a discrepancy
        #         T = copy.copy(C) # Save current connection polynomial
                
        #         # Calculate new connection polynomial C(x) = C(x) - d/b * x^m * B(x)
        #         # In GF(2), subtraction is the same as addition (XOR), and d/b is always 1 when d and b are 1
        #         # So C(x) = C(x) + x^m * B(x)
        #         while len(C) <= len(B) + m:
        #             C.append(0) # Extend C to fit the new degree
        #         for j, coeff in enumerate(B):
        #             C[j + m] ^= coeff # add B(x) shifted by m
                
        #         if 2 * L <= n:                    
        #             B = copy.copy(T)
        #             L = n + 1 - L
        #             b = d
        #             m = 1
        #         else:
        #             m += 1
        #     else:
        #         m += 1
        
        # return C
    
    @staticmethod
    def min_poly(sequence):
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