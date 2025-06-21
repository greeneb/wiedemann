import numpy as np
from berlekamp_massey import bm
# from gf2_linalg import GF2Vector, GF2Matrix
from gf2 import gf2matrix

class wiedemann:
    """
    Implementation of Wiedemann's algorithm for solving linear systems over GF(2).
    This class provides methods to solve the equation Mw = 0 or Ax = b using Wiedemann's algorithm.
    """
    
    @staticmethod
    def solve(M, max_iter=None, verbose=True):
        """
        Wiedemann's algorithm for solving Mw = 0 over GF(2)
        
        Inputs:
            M: n x n numpy array over GF(2)
            max_iter: maximum number of iterations (default: 2n)
        
        Returns:
            w: n-dimensional numpy array over GF(2) such that Mw = 0, or None if no solution found
        """
        if isinstance(M, np.ndarray):
            M = gf2matrix.from_dense(M)
        
        n = M.n_rows

        if max_iter is None:
            max_iter = 2 * M.n_rows
        
        for attempt in range(1, max_iter + 1):
            if verbose: print(f"\nAttempt {attempt} with a different random vector x_base:")
            
            # Generate a random vector u and build proper Krylov sequence S = [u, M·u, M^2·u, ...]
            x_base = np.random.randint(0, 2, n, dtype=np.int8)
            x = M.apply(x_base)
            S = [x.copy()]
            for _ in range(2 * n - 1):
                S.append(M.apply(S[-1]))
            print(x_base)
            print(S)

            # Generate the sequence S_y = [y^T x, y^T M x, ...]
            y = np.random.randint(0, 2, n, dtype=np.int8)
            S_y = [np.bitwise_xor.reduce(y & S_i) for S_i in S]

            # Apply Berlekamp-Massey to find the minimal polynomial
            q = bm.find_minimal_polynomial(S_y)
            print(S_y)
            print(q)
            
            if verbose: print(f"Minimal polynomial: {q}") # TODO: polynomial tostring
            
            # If the minimal polynomial is trivial, continue to the next attempt
            if len(q) <= 1:
                if verbose: print("Trivial minimal polynomial, trying another y vector.")
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
                if verbose: print("Minimal polynomial does not annihilate S, trying another y vector.")
                continue

            # Output a kernel vector using m(M)·u = 0 => w = ∑_{i=0}^d q[i]·S[i]
            # since q[0] == 1, start with S[0]
            w = S[0].copy()
            for i in range(1, d+1):
                if q[i]:
                    w = np.bitwise_xor(w, S[i])

            # Verify Mw = 0
            Mw = M.apply(w)
            if verbose: 
                print("Kernel vector w:")
                print(w)
                print("Verification: M * w =")
                print(Mw)
                print("Is zero vector:", not np.any(Mw))

            if not np.any(Mw):
                if verbose: print(f"Success on attempt {attempt}!")
                return w


    def wiedemann(A, b, max_iter=None):
        """
        Wiedemann's algorithm for solving Ax = b over GF(2).
        A: n x n numpy array over GF(2)
        b: n-dimensional numpy array over GF(2)
        max_iter: maximum number of iterations (default: 2n)
        Returns: x such that Ax = b over GF(2), or None if no solution found
        """
        n = A.shape[0]
        if max_iter is None:
            max_iter = 2 * n

        # Choose a random vector c in GF(2)^n
        c = np.random.randint(0, 2, size=n, dtype=np.uint8)

        # Generate the sequence s_k = c^T A^k b for k = 0, ..., max_iter-1
        s = []
        v = b.copy()
        for _ in range(max_iter):
            s.append(int(np.dot(c, v) % 2))
            v = gf2_matvec(A, v)

        # Find minimal polynomial using Berlekamp-Massey
        minimal_poly = berlekamp_massey(s)
        d = len(minimal_poly) - 1

        # Compute v_0, ..., v_{d-1}
        vs = [b.copy()]
        v = b.copy()
        for _ in range(1, d):
            v = gf2_matvec(A, v)
            vs.append(v.copy())

        # Solve for x: sum_{i=0}^{d-1} minimal_poly[i] * A^i b = 0
        # So, x = sum_{i=1}^{d} (-minimal_poly[i]) * A^{i-1} b
        x = np.zeros(n, dtype=np.uint8)
        for i in range(1, d + 1):
            coeff = minimal_poly[i] if i < len(minimal_poly) else 0
            if coeff:
                x ^= vs[i - 1]

        # Check if Ax = b
        if np.array_equal(gf2_matvec(A, x), b):
            return x
        else:
            return None