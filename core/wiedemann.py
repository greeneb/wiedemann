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
    def wiedemann_algorithm_1(A, b, max_passes=5, verbose=False):
        """
        Solve Ax = b over GF(2) using Wiedemann Algorithm 1.

        Args:
            A: gf2matrix object (assumed to be square and nonsingular)
            b: right-hand side vector (numpy array of 0s and 1s)
            max_passes: number of random trials before giving up
            verbose: print debug info

        Returns:
            x: solution vector such that A x = b
        """
        n = A.n_rows
        y = np.zeros(n, dtype=np.int8)  # initialize solution accumulator
        b_k = b.copy()
        d_k = 0

        for pass_num in range(max_passes):
            if verbose:
                print(f"\n--- Pass {pass_num+1} ---")

            if np.all(b_k == 0):
                if verbose:
                    print("b_k is zero vector; solution found.")
                return y

            u = np.random.randint(0, 2, size=n, dtype=np.int8)

            # Compute the sequence (u, A^i b_k) for i = 0 to 2(n - d_k) - 1
            seq_len = 2 * (n - d_k)
            s = []
            v = b_k.copy()
            for _ in range(seq_len):
                s.append(np.dot(u, v) % 2)  # dot product mod 2
                v = A.apply(v)

            f = bm.min_poly(s)
            deg_f = len(f) - 1

            # Evaluate f_minus(A) b_k and add to solution: y += f_minus(A) b_k
            # f_minus(z) = (f(z) - f(0)) / z
            f_minus = f[1:]  # skip constant term
            v = b_k.copy()
            acc = np.zeros(n, dtype=np.int8)
            for coeff in reversed(f_minus):  # f[1]*A^0 b + f[2]*A^1 b + ...
                if coeff:
                    acc ^= v
                v = A.apply(v)
            y ^= acc  # y_{k+1} = y_k + f_minus(A) b_k

            # b_{k+1} = b_k + A y
            b_k = b_k ^ A.apply(y)
            d_k += deg_f

            if verbose:
                print(f"Minimal polynomial (deg {deg_f}): {f}")
                print(f"Accumulated solution y: {y}")
                print(f"Next residual b_k: {b_k}")
                print(f"Total degree d_k: {d_k}")
        
        raise ValueError("Failed to find solution after max_passes")

    @staticmethod
    def wiedemann_algorithm_2(A, b, max_passes=5, verbose=False):
        """
        Solve Ax = b over GF(2) using Wiedemann Algorithm 2.

        Args:
            A: gf2matrix object (assumed to be square and nonsingular)
            b: right-hand side vector (numpy array of 0s and 1s)
            max_passes: number of random trials before giving up
            verbose: print debug info

        Returns:
            x: solution vector such that A x = b
        """
        n = A.n_rows
        assert A.n_cols == n, "Matrix A must be square"
        assert len(b) == n, "Vector b must have the same length as the number of rows in A"
        
        # Step 1: Precompute and store A^i b for i in 0..2n-1
        if verbose: print("Precomputing A^i b for i in 0..2n-1")
        powers = [b.copy()]
        for _ in range(2 * n - 1):
            powers.append(A.apply(powers[-1]))
            
        # Step 2: Initializeaiton 
        k = 0
        gk = [1] # g_0(z) = 1
        
        while k < n and len(gk) < n + 1:
            if verbose: print(f"\n--- Iteration {k + 1} ---")
            
            uk = np.zeros(n, dtype=np.int8)
            uk[k] = 1
            
            # Step 3: Form sequence s = (uk, A^i b) = uk * powers[i]
            s = [np.dot(uk, powers[i]) % 2 for i in range(2 * n)]
            
            # Step 4: Apply gk(z) to s
            s_applied = wiedemann.apply_poly_to_sequence(gk, s)
            
            # Step 5: Compute minimal polynomial fk+1
            fk1 = bm.min_poly(s_applied)
            if verbose:
                print(f"u_{k+1} = {uk}")
                print(f"g_{k}(z): {gk}")
                print(f"Sequence after applying g_k: {s_applied}")
                print(f"Minimal polynomial f_{k+1}(z): {fk1}")
                
            # Step 6: Update gk
            gk = wiedemann.poly_mul(fk1, gk)
            
            k += 1
        
        # Step 7: compute x using f = g_k and powers of A^i b
        f = gk
        d = len(f) - 1
        
        x = np.zeros(n, dtype=np.int8)
        for i in range(d + 1):
            if f[i]:
                x ^= powers[i - 1]
                
        if verbose:
            print(f"Final polynomial g_{k}(z): {f}")
            print(f"Solution x: {x}")
        
        return x
    
    @staticmethod
    def apply_poly_to_sequence(poly, sequence):
        """
        Multiply a polynomial (represented by coefficient list) with a sequence.
        Polynomial is applied as: conv(poly, sequence), truncated appropriately.
        
        Args:
            poly: list of GF(2) coefficients, lowest degree first (e.g. [1, 0, 1] = 1 + z^2)
            sequence: list of elements over GF(2)

        Returns:
            List of GF(2) values: the result of polynomial applied to the sequence
        """
        result_len = len(sequence) - len(poly) + 1
        result = []
        for i in range(result_len):
            acc = 0
            for j, coeff in enumerate(poly):
                if i + j < len(sequence):
                    acc ^= coeff * sequence[i + j]
            result.append(acc % 2) # TODO: maybe mod 2?
        return result
    
    @staticmethod
    def poly_mul(p1, p2):
        """
        Multiply two polynomials over GF(2).
        Args:
            p1: list of coefficients for the first polynomial (lowest degree first)
            p2: list of coefficients for the second polynomial (lowest degree first)
            
        Returns:
            List of coefficients for the product polynomial (lowest degree first)
        """
        result = [0] * (len(p1) + len(p2) - 1)
        for i, coeff1 in enumerate(p1):
            if coeff1 == 0:
                continue
            for j, coeff2 in enumerate(p2):
                if coeff2:
                    result[i + j] ^= 1
        return result
        
    # @staticmethod
    # def solve(M, max_iter=None, verbose=True):
    #     """
    #     Wiedemann's algorithm for solving Mw = 0 over GF(2)
        
    #     Inputs:
    #         M: n x n numpy array over GF(2)
    #         max_iter: maximum number of iterations (default: 2n)
        
    #     Returns:
    #         w: n-dimensional numpy array over GF(2) such that Mw = 0, or None if no solution found
    #     """
    #     if isinstance(M, np.ndarray):
    #         M = gf2matrix.from_dense(M)
        
    #     n = M.n_rows

    #     if max_iter is None:
    #         max_iter = 2 * M.n_rows
        
    #     for attempt in range(1, max_iter + 1):
    #         if verbose: print(f"\nAttempt {attempt} with a different random vector x_base:")
            
    #         # Generate a random vector u and build proper Krylov sequence S = [u, M·u, M^2·u, ...]
    #         x_base = np.random.randint(0, 2, n, dtype=np.int8)
    #         x = M.apply(x_base)
    #         S = [x.copy()]
    #         for _ in range(2 * n - 1):
    #             S.append(M.apply(S[-1]))
    #         print(x_base)
    #         print(S)

    #         # Generate the sequence S_y = [y^T x, y^T M x, ...]
    #         y = np.random.randint(0, 2, n, dtype=np.int8)
    #         S_y = [np.bitwise_xor.reduce(y & S_i) for S_i in S]

    #         # Apply Berlekamp-Massey to find the minimal polynomial
    #         q = bm.find_minimal_polynomial(S_y)
    #         print(S_y)
    #         print(q)
            
    #         if verbose: print(f"Minimal polynomial: {q}") # TODO: polynomial tostring
            
    #         # If the minimal polynomial is trivial, continue to the next attempt
    #         if len(q) <= 1:
    #             if verbose: print("Trivial minimal polynomial, trying another y vector.")
    #             continue
            
    #         # Check if minimal polynomial annihilates scalar sequence S_y
    #         annihilates = True
    #         d = len(q) - 1
    #         for k in range(len(S_y) - d):
    #             acc = 0
    #             for i in range(d + 1):
    #                 if q[i]:
    #                     acc ^= S_y[k + i]
    #             if acc != 0:
    #                 annihilates = False
    #                 break

    #         if not annihilates:
    #             if verbose: print("Minimal polynomial does not annihilate S, trying another y vector.")
    #             continue

    #         # Output a kernel vector using m(M)·u = 0 => w = ∑_{i=0}^d q[i]·S[i]
    #         # since q[0] == 1, start with S[0]
    #         w = S[0].copy()
    #         for i in range(1, d+1):
    #             if q[i]:
    #                 w = np.bitwise_xor(w, S[i])

    #         # Verify Mw = 0
    #         Mw = M.apply(w)
    #         if verbose: 
    #             print("Kernel vector w:")
    #             print(w)
    #             print("Verification: M * w =")
    #             print(Mw)
    #             print("Is zero vector:", not np.any(Mw))

    #         if not np.any(Mw):
    #             if verbose: print(f"Success on attempt {attempt}!")
    #             return w


    # def wiedemann(A, b, max_iter=None):
    #     """
    #     Wiedemann's algorithm for solving Ax = b over GF(2).
    #     A: n x n numpy array over GF(2)
    #     b: n-dimensional numpy array over GF(2)
    #     max_iter: maximum number of iterations (default: 2n)
    #     Returns: x such that Ax = b over GF(2), or None if no solution found
    #     """
    #     n = A.shape[0]
    #     if max_iter is None:
    #         max_iter = 2 * n

    #     # Choose a random vector c in GF(2)^n
    #     c = np.random.randint(0, 2, size=n, dtype=np.uint8)

    #     # Generate the sequence s_k = c^T A^k b for k = 0, ..., max_iter-1
    #     s = []
    #     v = b.copy()
    #     for _ in range(max_iter):
    #         s.append(int(np.dot(c, v) % 2))
    #         v = gf2_matvec(A, v)

    #     # Find minimal polynomial using Berlekamp-Massey
    #     minimal_poly = berlekamp_massey(s)
    #     d = len(minimal_poly) - 1

    #     # Compute v_0, ..., v_{d-1}
    #     vs = [b.copy()]
    #     v = b.copy()
    #     for _ in range(1, d):
    #         v = gf2_matvec(A, v)
    #         vs.append(v.copy())

    #     # Solve for x: sum_{i=0}^{d-1} minimal_poly[i] * A^i b = 0
    #     # So, x = sum_{i=1}^{d} (-minimal_poly[i]) * A^{i-1} b
    #     x = np.zeros(n, dtype=np.uint8)
    #     for i in range(1, d + 1):
    #         coeff = minimal_poly[i] if i < len(minimal_poly) else 0
    #         if coeff:
    #             x ^= vs[i - 1]

    #     # Check if Ax = b
    #     if np.array_equal(gf2_matvec(A, x), b):
    #         return x
    #     else:
    #         return None