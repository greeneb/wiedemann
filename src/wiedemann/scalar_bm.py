"""
Berlekamp-Massey Algorithm Implementation

This module implements the Berlekamp-Massey algorithm for finding the minimal
polynomial of a linearly recurrent sequence over finite fields. This algorithm
is a key component of the Wiedemann algorithm for solving linear systems.

The Berlekamp-Massey algorithm finds the shortest linear feedback shift register
(LFSR) that generates the given sequence, which corresponds to the minimal polynomial.

References:
- Berlekamp, E. R. (1968). Algebraic coding theory.
- Massey, J. L. (1969). Shift-register synthesis and BCH decoding.
"""

import numpy as np
import galois

def berlekamp_massey(sequence, field):
    """
    Berlekamp-Massey algorithm over GF(p).
    
    This implementation follows the standard algorithm as described in:
    - Berlekamp, E. R. (1968). Algebraic coding theory.
    - Massey, J. L. (1969). Shift-register synthesis and BCH decoding.
    
    Args:
        sequence (list or array-like): Sequence of field elements.
        field (galois.GF): Galois field class.
        
    Returns:
        galois.Poly: Minimal polynomial of the sequence.
    """
    n = len(sequence)
    if n == 0:
        return galois.Poly([1], field=field)
    
    seq = field(sequence)
    
    # Initialize
    C = field([1])  # Current LFSR polynomial
    B = field([1])  # Backup polynomial
    L = 0           # Current LFSR length
    m = 1           # Number of iterations since last update
    b = field(1)    # Last non-zero discrepancy
    
    for i in range(n):
        # Compute discrepancy
        d = seq[i]
        for j in range(1, min(L + 1, len(C))):
            d += C[j] * seq[i - j]
            
        if d == 0:
            m += 1
            continue
        
        # Save current polynomial
        T = C.copy()
        
        # Update polynomial
        factor = d / b
        
        # Ensure C has enough coefficients
        while len(C) < len(B) + m:
            C = field(np.concatenate([C, [field(0)]]))
        
        # Update coefficients
        for j in range(len(B)):
            if j + m < len(C):
                C[j + m] -= factor * B[j]
        
        # Update LFSR length and backup polynomial
        if 2 * L <= i:
            L = i + 1 - L
            B = T
            b = d
            m = 1
        else:
            m += 1
    
    # Normalize the polynomial (make it monic)
    if len(C) > 0 and C[-1] != 0:
        # Find the leading coefficient
        lead_coeff = C[-1]
        if lead_coeff != 1:
            C = C / lead_coeff
    
    # Convert to polynomial (coefficients in reverse order for galois.Poly)
    return galois.Poly(C[::-1], field=field)