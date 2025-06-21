import numpy as np

from gf2 import gf2matrix
from berlekamp_massey import berlekamp_massey

class ScalarWiedemann:
    """
    Implementation of the scalar Wiedemann algorithm for solving sparse linear systems
    over GF(2).
    """
    
    @staticmethod
    def solve(M, max_attempts=5):
        """
        Solve the linear system Mw=0 using Wiedemann's algorithm.
        https://en.wikipedia.org/wiki/Block_Wiedemann_algorithm
        
        Args:
            M: The coefficient matrix
            max_attempts: Maximum number of attempts with different random vectors
            
        Returns:
            The solution vector w, or None if no solution is found
        """
        if isinstance(M, np.ndarray):
            M = gf2matrix.from_dense(M)
        
        n = M.n_rows
        
        for attempt in range(1, max_attempts + 1):
            print(f"\nAttempt {attempt} with a different random vector u:")
            
            # Generate a random vector u and build proper Krylov sequence S = [u, M·u, M^2·u, ...]
            u = np.random.randint(0, 2, n, dtype=np.int8)
            S = [u.copy()]
            for _ in range(2 * n - 1):
                S.append(M.apply(S[-1]))

            # Generate the sequence S_y = [y^T x, y^T M x, ...]
            y = np.random.randint(0, 2, n, dtype=np.int8)
            S_y = [np.bitwise_xor.reduce(y & S_i) for S_i in S]

            # Apply Berlekamp-Massey to find the minimal polynomial
            q = berlekamp_massey.find_minimal_polynomial(S_y)
            
            print(f"Minimal polynomial: {q}") # TODO: polynomial tostring
            
            # If the minimal polynomial is trivial, continue to the next attempt
            if len(q) <= 1:
                print("Trivial minimal polynomial, trying another y vector.")
                continue
            
            # Check if minimal polynomial annihilates scalar sequence S_y
            annihilates = True
            d = len(q) - 1
            for k in range(len(S_y) - d):
                acc = 0
                for i in range(d + 1):
                    if q[i]:
                        acc ^= S_y[k + i]
                if acc != 0:
                    annihilates = False
                    break

            if not annihilates:
                print("Minimal polynomial does not annihilate S, trying another y vector.")
                continue

            # Output a kernel vector using m(M)·u = 0 => w = ∑_{i=0}^d q[i]·S[i]
            # since q[0] == 1, start with S[0]
            w = S[0].copy()
            for i in range(1, d+1):
                if q[i]:
                    w = np.bitwise_xor(w, S[i])

            # Verify Mw = 0
            Mw = M.apply(w)
            print("Kernel vector w:")
            print(w)
            print("Verification: M * w =")
            print(Mw)
            print("Is zero vector:", not np.any(Mw))

            if not np.any(Mw):
                print(f"Success on attempt {attempt}!")
                return w