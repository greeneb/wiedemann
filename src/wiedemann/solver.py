import galois
import numpy as np
from wiedemann.block_wiedemann import block_wiedemann
from wiedemann.block_bm import null_space_mod_p
from wiedemann.scalar_wiedemann import solve as scalar_wiedemann_solve


def solve_homogeneous(A, field, m=2, max_iter=None, max_retries=5):
    """
    Solve the homogeneous system A x = 0 using Block Wiedemann with explicit reconstruction.

    Args:
        A (galois.FieldArray): n x n matrix over GF(p).
        field (galois.GF): Field definition.
        m (int): Block size.
        max_iter (int, optional): Number of iterations. Default: 2*n.

    Returns:
        galois.FieldArray: A nonzero kernel vector x with A x = 0.
    """
    n = A.shape[0]

    for attempt in range(max_retries):
        # Run Block Wiedemann + BBM
        Q, U, V = block_wiedemann(A, field, m=m, max_iter=max_iter)
        d = len(Q) - 1

        # Build Krylov blocks: [V, A V, ..., A^d V]
        krylov = [V]
        for _ in range(1, d + 1):
            krylov.append(A @ krylov[-1])

        # Reconstruct Y = sum_{i=0..d} (A^i V) Q_i (n x m)
        Y = field.Zeros((n, m))
        for i in range(d + 1):
            Y += krylov[i] @ Q[i]

        # Try to find non-zero z with A Y z = 0, then x = Y z is a candidate
        AY = A @ Y
        z = null_space_mod_p(AY, field)
        if z is not None:
            z = field(np.array(z).reshape(-1, 1))
            x = Y @ z
            x = field(x.reshape(-1))
            if not np.all(x == 0) and (A @ x == 0).all():
                return x

    # Fallback: use scalar Wiedemann to produce a kernel vector (still Wiedemann-based)
    # try:
    #     x = scalar_wiedemann_solve(A, field)
    #     if not np.all(x == 0) and (A @ x == 0).all():
    #         return x
    # except Exception:
    #     pass

    raise ValueError("Failed to find nonzero kernel vector.")


def solve_inhomogeneous(A, b, field, m=2, max_iter=None):
    """
    Attempt to solve A x = b using Block Wiedemann.
    Currently not implemented (requires extended reconstruction method).
    """
    raise NotImplementedError("Inhomogeneous solver not yet implemented.")