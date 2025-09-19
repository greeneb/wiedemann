#!/usr/bin/env python3
"""
Simple performance comparison between Wiedemann and Gaussian elimination.
"""

import time
import galois
import numpy as np
from wiedemann.scalar_wiedemann import solve as scalar_solve
from wiedemann.block_wiedemann import block_wiedemann

def gaussian_elimination_nullspace(A):
    """Find nullspace using Gaussian elimination."""
    A_np = A.view(np.ndarray)
    U, s, Vh = np.linalg.svd(A_np)
    nullspace_indices = np.where(np.abs(s) < 1e-10)[0]
    if len(nullspace_indices) > 0:
        return A.field(nullspace_indices[0])
    return None

def create_test_matrix(n, field_size=7, density=0.1):
    """Create a test matrix."""
    GF = galois.GF(field_size)
    
    # Create a random matrix
    A = np.random.randint(0, field_size, (n, n))
    
    # Make it sparse by zeroing out most entries
    mask = np.random.random((n, n)) > density
    A[mask] = 0
    
    # Make it singular
    if n > 1:
        A[-1, :] = (A[0, :] + A[1, :]) % field_size
    
    return GF(A)

def benchmark_single_matrix(A, name):
    """Benchmark a single matrix."""
    print(f"\n=== {name} ===")
    print(f"Matrix shape: {A.shape}")
    density = np.count_nonzero(A.view(np.ndarray)) / A.size
    print(f"Matrix density: {density:.3f}")
    
    # Gaussian elimination
    start_time = time.time()
    try:
        kernel_ge = gaussian_elimination_nullspace(A)
        ge_time = time.time() - start_time
        print(f"Gaussian Elimination: {ge_time:.4f}s")
        if kernel_ge is not None:
            print(f"  Found kernel vector: {kernel_ge[:5]}...")
    except Exception as e:
        print(f"Gaussian Elimination failed: {e}")
        ge_time = float('inf')
    
    # Scalar Wiedemann
    start_time = time.time()
    try:
        kernel_sw = scalar_solve(A, A.field)
        sw_time = time.time() - start_time
        print(f"Scalar Wiedemann: {sw_time:.4f}s")
        print(f"  Found kernel vector: {kernel_sw[:5]}...")
    except Exception as e:
        print(f"Scalar Wiedemann failed: {e}")
        sw_time = float('inf')
    
    # Block Wiedemann
    start_time = time.time()
    try:
        kernel_bw = block_wiedemann(A, A.field, m=min(3, A.shape[0]//2), n=min(3, A.shape[0]//2))
        bw_time = time.time() - start_time
        print(f"Block Wiedemann: {bw_time:.4f}s")
        print(f"  Found kernel vector: {kernel_bw[:5]}...")
    except Exception as e:
        print(f"Block Wiedemann failed: {e}")
        bw_time = float('inf')
    
    return ge_time, sw_time, bw_time

def main():
    print("Performance Comparison: Wiedemann vs Gaussian Elimination")
    print("=" * 60)
    
    # Test cases: (size, density, name)
    test_cases = [
        (20, 0.3, "Small Dense Matrix"),
        (20, 0.05, "Small Sparse Matrix"),
        (50, 0.2, "Medium Dense Matrix"),
        (50, 0.02, "Medium Sparse Matrix"),
        (100, 0.1, "Large Dense Matrix"),
        (100, 0.01, "Large Sparse Matrix"),
    ]
    
    results = []
    
    for n, density, name in test_cases:
        A = create_test_matrix(n, density=density)
        ge_time, sw_time, bw_time = benchmark_single_matrix(A, name)
        results.append((name, ge_time, sw_time, bw_time))
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for name, ge_time, sw_time, bw_time in results:
        print(f"\n{name}:")
        times = [("Gaussian Elimination", ge_time), ("Scalar Wiedemann", sw_time), ("Block Wiedemann", bw_time)]
        valid_times = [(method, t) for method, t in times if t != float('inf')]
        
        if valid_times:
            fastest = min(valid_times, key=lambda x: x[1])
            print(f"  Fastest: {fastest[0]} ({fastest[1]:.4f}s)")
            
            for method, time_val in valid_times:
                speedup = fastest[1] / time_val if time_val > 0 else float('inf')
                print(f"  {method}: {time_val:.4f}s (speedup: {speedup:.2f}x)")
        else:
            print("  All methods failed")

if __name__ == "__main__":
    main()
