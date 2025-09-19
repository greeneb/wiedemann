#!/usr/bin/env python3
"""
Performance comparison between Wiedemann algorithms and Gaussian elimination.
"""

import time
import galois
import numpy as np
from scipy.sparse import random as sparse_random
from scipy.sparse import csr_matrix
from wiedemann.scalar_wiedemann import solve as scalar_solve
from wiedemann.block_wiedemann import block_wiedemann

def create_sparse_matrix(n, density=0.1, field_size=7):
    """Create a sparse singular matrix."""
    GF = galois.GF(field_size)
    
    # Create a random sparse matrix
    sparse_mat = sparse_random(n, n, density=density, format='csr', random_state=42)
    A_sparse = sparse_mat.toarray()
    
    # Make it singular by setting the last row to be a linear combination of others
    if n > 1:
        A_sparse[-1, :] = (A_sparse[0, :] + A_sparse[1, :]) % field_size
    
    return GF(A_sparse.astype(int))

def create_dense_matrix(n, field_size=7):
    """Create a dense singular matrix."""
    GF = galois.GF(field_size)
    
    # Create a random matrix
    A = np.random.randint(0, field_size, (n, n))
    
    # Make it singular
    if n > 1:
        A[-1, :] = (A[0, :] + A[1, :]) % field_size
    
    return GF(A)

def gaussian_elimination_nullspace(A):
    """Find nullspace using Gaussian elimination."""
    A_np = A.view(np.ndarray)
    U, s, Vh = np.linalg.svd(A_np)
    nullspace_indices = np.where(np.abs(s) < 1e-10)[0]
    if len(nullspace_indices) > 0:
        return A.field(nullspace_indices[0])
    return None

def benchmark_methods(A, name):
    """Benchmark different methods on matrix A."""
    print(f"\n=== {name} ===")
    print(f"Matrix shape: {A.shape}")
    print(f"Matrix density: {np.count_nonzero(A.view(np.ndarray)) / A.size:.3f}")
    
    results = {}
    
    # Gaussian elimination
    start_time = time.time()
    try:
        kernel_ge = gaussian_elimination_nullspace(A)
        ge_time = time.time() - start_time
        results['Gaussian Elimination'] = ge_time
        print(f"Gaussian Elimination: {ge_time:.4f}s")
    except Exception as e:
        print(f"Gaussian Elimination failed: {e}")
        results['Gaussian Elimination'] = float('inf')
    
    # Scalar Wiedemann
    start_time = time.time()
    try:
        kernel_sw = scalar_solve(A, A.field)
        sw_time = time.time() - start_time
        results['Scalar Wiedemann'] = sw_time
        print(f"Scalar Wiedemann: {sw_time:.4f}s")
    except Exception as e:
        print(f"Scalar Wiedemann failed: {e}")
        results['Scalar Wiedemann'] = float('inf')
    
    # Block Wiedemann
    start_time = time.time()
    try:
        kernel_bw = block_wiedemann(A, A.field, m=min(4, A.shape[0]//2), n=min(4, A.shape[0]//2))
        bw_time = time.time() - start_time
        results['Block Wiedemann'] = bw_time
        print(f"Block Wiedemann: {bw_time:.4f}s")
    except Exception as e:
        print(f"Block Wiedemann failed: {e}")
        results['Block Wiedemann'] = float('inf')
    
    return results

def main():
    print("Performance Comparison: Wiedemann vs Gaussian Elimination")
    print("=" * 60)
    
    # Test different matrix sizes and densities
    test_cases = [
        (50, 0.1, "Small Sparse Matrix"),
        (50, 0.8, "Small Dense Matrix"),
        (100, 0.05, "Medium Sparse Matrix"),
        (100, 0.5, "Medium Dense Matrix"),
        (200, 0.02, "Large Sparse Matrix"),
        (200, 0.3, "Large Dense Matrix"),
    ]
    
    all_results = {}
    
    for n, density, name in test_cases:
        if density < 0.5:  # Sparse
            A = create_sparse_matrix(n, density)
        else:  # Dense
            A = create_dense_matrix(n)
        
        results = benchmark_methods(A, name)
        all_results[name] = results
    
    # Summary
    print("\n" + "=" * 60)
    print("PERFORMANCE SUMMARY")
    print("=" * 60)
    
    for name, results in all_results.items():
        print(f"\n{name}:")
        fastest = min(results.items(), key=lambda x: x[1])
        print(f"  Fastest: {fastest[0]} ({fastest[1]:.4f}s)")
        
        for method, time_val in results.items():
            if time_val != float('inf'):
                speedup = fastest[1] / time_val if time_val > 0 else float('inf')
                print(f"  {method}: {time_val:.4f}s (speedup: {speedup:.2f}x)")

if __name__ == "__main__":
    main()
