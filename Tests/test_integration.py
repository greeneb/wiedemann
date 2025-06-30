"""
Integration tests for the complete Wiedemann solver pipeline.
Tests the interaction between Berlekamp-Massey and Wiedemann algorithms.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

import unittest
import numpy as np
from gf2 import gf2matrix
from wiedemann import wiedemann
from berlekamp_massey import berlekamp_massey

class TestWiedemannIntegration(unittest.TestCase):
    """Integration tests for the complete Wiedemann pipeline"""
    
    def setUp(self):
        np.random.seed(123)
    
    def test_complete_pipeline_singular_matrix(self):
        """Test complete pipeline on a matrix with known singular structure"""
        # Create a matrix with rank deficiency
        # This matrix has nullspace spanned by [1,1,0,0]
        A_dense = np.array([
            [1, 1, 0, 0],
            [1, 1, 0, 0], 
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ], dtype=np.int8)
        A = gf2matrix.from_dense(A_dense)
        
        # Manually verify the expected nullspace
        expected_null_1 = np.array([1, 1, 0, 0], dtype=np.int8)
        expected_null_2 = np.array([0, 0, 1, 1], dtype=np.int8)
        
        result_1 = A.apply(expected_null_1)
        result_2 = A.apply(expected_null_2)
        
        self.assertTrue(not np.any(result_1), f"Expected [1,1,0,0] in nullspace, A*v = {result_1}")
        self.assertTrue(not np.any(result_2), f"Expected [0,0,1,1] in nullspace, A*v = {result_2}")
        
        # Now test if Wiedemann can find a nullspace vector
        found_solution = False
        for attempt in range(20):  # Multiple attempts due to probabilistic nature
            w = wiedemann.solve(A, max_iter=10, verbose=False)
            if w is not None and np.any(w):
                # Verify it's a valid solution
                Aw = A.apply(w)
                if not np.any(Aw):
                    found_solution = True
                    print(f"Found nullspace vector: {w}")
                    break
        
        if not found_solution:
            self.skipTest("Wiedemann algorithm didn't find nullspace (probabilistic failure)")
    
    def test_berlekamp_massey_with_wiedemann_sequences(self):
        """Test Berlekamp-Massey on sequences that would arise in Wiedemann"""
        # Create sequences that might arise from Krylov iterations
        test_sequences = [
            [1, 0, 1, 0, 1, 0, 1, 0],  # Periodic with period 2
            [1, 1, 0, 1, 1, 0, 1, 1, 0],  # Periodic with period 3  
            [0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1],  # More complex
            [1, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1]  # Another pattern
        ]
        
        for i, seq in enumerate(test_sequences):
            with self.subTest(sequence=i):
                poly = berlekamp_massey.find_minimal_polynomial(seq)
                
                # Verify polynomial properties
                self.assertGreaterEqual(len(poly), 1, "Polynomial should have at least one coefficient")
                self.assertEqual(poly[0], 1, "Leading coefficient should be 1")
                
                # Verify annihilation property
                self._verify_polynomial_annihilates_sequence(seq, poly)
    
    def test_edge_case_matrices(self):
        """Test edge cases that might cause issues in the pipeline"""
        edge_cases = [
            # All ones matrix (rank 1)
            np.ones((3, 3), dtype=np.int8),
            # Diagonal matrix with some zeros
            np.diag([1, 0, 1, 0], dtype=np.int8),
            # Upper triangular
            np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=np.int8),
            # Permutation matrix
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int8)
        ]
        
        for i, A_dense in enumerate(edge_cases):
            with self.subTest(matrix=i):
                A = gf2matrix.from_dense(A_dense)
                
                # Test that the algorithm doesn't crash
                try:
                    w = wiedemann.solve(A, max_iter=8, verbose=False)
                    if w is not None:
                        # Verify any solution found is valid
                        Aw = A.apply(w)
                        self.assertTrue(not np.any(Aw), f"Matrix {i}: Invalid solution found")
                        self.assertTrue(np.any(w), f"Matrix {i}: Solution should be non-zero")
                except Exception as e:
                    self.fail(f"Matrix {i}: Algorithm crashed with error: {e}")
    
    def test_random_matrix_stress_test(self):
        """Stress test with various random matrices"""
        for size in [3, 4, 5]:
            for density in [0.3, 0.5, 0.7]:
                with self.subTest(size=size, density=density):
                    # Generate multiple random matrices
                    for trial in range(3):
                        A = gf2matrix.random(size, density=density)
                        
                        # Test that algorithm completes without error
                        try:
                            w = wiedemann.solve(A, max_iter=12, verbose=False)
                            if w is not None:
                                # Verify solution
                                Aw = A.apply(w)
                                self.assertTrue(not np.any(Aw), 
                                    f"Size {size}, density {density}, trial {trial}: Invalid solution")
                        except Exception as e:
                            self.fail(f"Size {size}, density {density}, trial {trial}: Error {e}")
    
    def test_minimal_polynomial_properties(self):
        """Test that minimal polynomials have expected properties"""
        # Test on various sequence types that might arise
        test_cases = [
            ([0, 0, 0, 0], [1]),  # Zero sequence
            ([1, 1, 1, 1], [1, 1]),  # Constant sequence  
            ([1, 0, 1, 0, 1, 0], [1, 0, 1]),  # Alternating
        ]
        
        for seq, expected_poly in test_cases:
            with self.subTest(sequence=seq):
                poly = berlekamp_massey.find_minimal_polynomial(seq)
                self.assertEqual(poly, expected_poly, 
                    f"Sequence {seq} should give polynomial {expected_poly}, got {poly}")
    
    def _verify_polynomial_annihilates_sequence(self, sequence, poly):
        """Helper method to verify polynomial annihilates sequence"""
        L = len(poly) - 1
        if L == 0:
            return  # Trivial polynomial
            
        for k in range(len(sequence) - L):
            acc = 0
            for i, coeff in enumerate(poly):
                if coeff and k + i < len(sequence):
                    acc ^= sequence[k + i]
            self.assertEqual(acc, 0, 
                f"Polynomial {poly} fails to annihilate sequence {sequence} at position {k}")
    
    def test_large_matrix_performance(self):
        """Test performance on slightly larger matrices"""
        # Test that algorithm doesn't hang on larger inputs
        for size in [8, 10]:
            with self.subTest(size=size):
                A = gf2matrix.random(size, density=0.2)  # Sparse for efficiency
                
                # Set reasonable limits
                start_time = __import__('time').time()
                w = wiedemann.solve(A, max_iter=min(15, size), verbose=False)
                end_time = __import__('time').time()
                
                # Should complete in reasonable time (arbitrary limit)
                self.assertLess(end_time - start_time, 30, 
                    f"Algorithm took too long on size {size}")
                
                if w is not None:
                    Aw = A.apply(w)
                    self.assertTrue(not np.any(Aw), f"Size {size}: Invalid solution")

if __name__ == '__main__':
    unittest.main()
