import numpy as np
import galois

def berlekamp_massey(sequence, field):
    """
    Berlekamp-Massey algorithm over GF(p).
    
    Args:
        sequence (list or array-like): Sequence of field elements.
        field (galois.GF): Galois field class.
        
    Returns:
        galois.Poly: Minimal polynomial of the sequence.
    """
    n = len(sequence)
    seq = field(sequence)
    
    C = field([1])
    B = field([1])
    L = 0
    m = 1
    b = field(1)
    
    for i in range(n):
        # Compute discrepancy
        d = seq[i]
        for j in range(1, L + 1):
            d += C[j] * seq[i-j]
            
        if d == 0:
            m += 1
            continue
        
        T = C.copy()
        factor = d / b
        
        C_list = list(C)
        B_list = list(B)
        
        if len(C_list) < len(B_list) + m:
            C_list += [field(0)] * (len(B_list) + m - len(C_list))
        
        for j in range(len(B_list)):
            C_list[j+m] -= factor * B_list[j]
        
        C = field(C_list)
        
        # Extend C if needed -- this doesn't work, the above seems to work better
        # C = np.pad(C, (0, m), constant_values=0)
        # C[m:m+len(B)] -= factor * B
        
        if 2 * L <= i:
            L = i + 1 - L
            B = T
            b = d
            m = 1
        else:
            m+= 1
            
    # if len(C) > 0 and C[-1] != 1:
    #     lead_inv = field(1) / C[-1]
    #     C = C * lead_inv
    
    # Convert to polynomial with C as coefficients
    return galois.Poly(C[::-1], field=field)