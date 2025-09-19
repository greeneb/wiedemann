#!/usr/bin/env python3
"""
Basic usage examples for the Wiedemann algorithm implementations.
"""

import galois
import numpy as np
from wiedemann.scalar_wiedemann import solve as scalar_solve, scalar_wiedemann
from wiedemann.block_wiedemann import block_wiedemann
from wiedemann.scalar_bm import berlekamp_massey

def example_scalar_wiedemann():
    """Example of using the scalar Wiedemann algorithm."""
    print("=== Scalar Wiedemann Algorithm Example ===")
    
    # Create a singular matrix over GF(7)
    GF = galois.GF(7)
    A = GF([[1, 2, 3],
            [2, 4, 6],  # This row is 2 * first row
            [1, 1, 1]])
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix rank: {np.linalg.matrix_rank(A.view(np.ndarray))}")
    
    # Find a kernel vector
    kernel_vector = scalar_solve(A, GF)
    print(f"Kernel vector: {kernel_vector}")
    print(f"A @ kernel_vector = {A @ kernel_vector}")
    print(f"Is kernel vector valid? {(A @ kernel_vector == 0).all()}")
    print()

def example_block_wiedemann():
    """Example of using the block Wiedemann algorithm."""
    print("=== Block Wiedemann Algorithm Example ===")
    
    # Create a larger singular matrix
    GF = galois.GF(5)
    A = GF([[1, 2, 3, 4],
            [2, 4, 1, 3],
            [3, 1, 4, 2],
            [1, 2, 3, 4]])  # Duplicate of first row
    
    print(f"Matrix A:\n{A}")
    print(f"Matrix rank: {np.linalg.matrix_rank(A.view(np.ndarray))}")
    
    # Find a kernel vector using block Wiedemann
    kernel_vector = block_wiedemann(A, GF, m=2, n=2)
    print(f"Kernel vector: {kernel_vector}")
    print(f"A @ kernel_vector = {A @ kernel_vector}")
    print(f"Is kernel vector valid? {(A @ kernel_vector == 0).all()}")
    print()

def example_berlekamp_massey():
    """Example of using the Berlekamp-Massey algorithm."""
    print("=== Berlekamp-Massey Algorithm Example ===")
    
    GF = galois.GF(5)
    
    # Create a periodic sequence
    sequence = [1, 2, 4, 3, 1, 2, 4, 3, 1, 2]
    print(f"Sequence: {sequence}")
    
    # Find minimal polynomial
    poly = berlekamp_massey(sequence, GF)
    print(f"Minimal polynomial: {poly}")
    print(f"Polynomial degree: {poly.degree}")
    
    # Verify the polynomial annihilates the sequence
    coeffs = poly.coeffs
    d = poly.degree
    print("Verifying polynomial annihilates sequence:")
    for k in range(d, len(sequence)):
        val = GF(0)
        for j, c in enumerate(coeffs):
            val += c * sequence[k - d + j]
        print(f"  Position {k}: {val} (should be 0)")
    print()

def example_quantum_error_correction():
    """Example relevant to quantum error correction."""
    print("=== Quantum Error Correction Example ===")
    
    # Create a parity check matrix (simplified example)
    GF = galois.GF(2)
    H = GF([[1, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 1]])
    
    print(f"Parity check matrix H:\n{H}")
    
    # Find a syndrome (error pattern)
    error = GF([1, 0, 0, 0, 0, 0])  # Single bit error
    syndrome = H @ error
    print(f"Error pattern: {error}")
    print(f"Syndrome: {syndrome}")
    
    # Find kernel vectors of H (these represent undetectable errors)
    try:
        kernel_vector = scalar_solve(H, GF)
        print(f"Undetectable error pattern: {kernel_vector}")
        print(f"H @ kernel_vector = {H @ kernel_vector}")
    except ValueError:
        print("No undetectable error patterns found (good for error correction)")
    print()

if __name__ == "__main__":
    example_scalar_wiedemann()
    example_block_wiedemann()
    example_berlekamp_massey()
    example_quantum_error_correction()
