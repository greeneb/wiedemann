class BerlekampMassey:
    """
    Implementation of the Berlekamp-Massey algorithm for finding the minimal polynomial
    of a sequence over GF(2).
    """
    
    @staticmethod
    def find_minimal_polynomial(sequence):
        """
        Apply the Berlekamp-Massey algorithm to find the minimal polynomial of a sequence.
        
        Args:
            sequence: A list of binary values (0s and 1s)
            
        Returns:
            A list representing the minimal polynomial (connection polynomial)
        """
        n = len(sequence)
        C = [1]       # Current connection polynomial
        B = [1]       # Copy of last connection polynomial for which a discrepancy was found
        L = 0         # Length of current LFSR
        m = 1         # Position of last discrepancy
        b = 1         # Initial discrepancy
        
        for N in range(n):
            # Calculate discrepancy
            d = sequence[N]
            for i in range(1, L+1):
                if i < len(C) and C[i] == 1 and N-i >= 0:
                    d ^= sequence[N-i]
            
            if d == 1:  # If there is a discrepancy
                # Save current connection polynomial
                T = copy.copy(C)
                
                # Calculate new connection polynomial C(x) = C(x) - d/b * x^m * B(x)
                # In GF(2), subtraction is the same as addition (XOR), and d/b is always 1 when d and b are 1
                # So C(x) = C(x) + x^m * B(x)
                while len(C) <= m + len(B) - 1:
                    C.append(0)
                    
                for j in range(len(B)):
                    C[j+m] ^= B[j]
                    
                if 2*L <= N:
                    L = N + 1 - L
                    B = copy.copy(T)
                    m = 1
                    b = d
                else:
                    m += 1
            else:
                m += 1
        
        return C
