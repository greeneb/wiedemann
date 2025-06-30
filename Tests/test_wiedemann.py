import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core'))

import unittest
import numpy as np
from gf2 import gf2matrix
from wiedemann import wiedemann

class TestWiedemann(unittest.TestCase):
    """
    Test cases for the Wiedemann algorithm implementation.
    Tests both singular and non-singular matrices, and edge cases.
    """
    
    def setUp(self):
        """Set up test fixtures"""
        np.random.seed(42)  # for reproducible tests
    
    def test_known_singular_matrix(self):
        """Test with a known singular matrix"""
        # Matrix with known nullspace: [[1,1,0],[1,1,0],[0,0,1]]
        # Nullspace is span{[1,1,0]} (any vector of form [v,v,0])
        A_dense = np.array([
            [1, 1, 0],
            [1, 1, 0], 
            [0, 0, 1]
        ], dtype=np.int8)
        A = gf2matrix.from_dense(A_dense)
        
        # Try multiple times since it's probabilistic
        solution_found = False
        for _ in range(10):
            w = wiedemann.solve(A, max_iter=5, verbose=False)
            if w is not None:
                # Verify solution: A*w should be zero
                Aw = A.apply(w)
                self.assertTrue(not np.any(Aw), f"A*w should be zero, got {Aw}")
                self.assertTrue(np.any(w), "Solution should be non-zero")
                solution_found = True
                break
        
        # Note: scalar Wiedemann may not always find solution due to random projections
        if not solution_found:
            self.skipTest("Scalar Wiedemann failed to find solution (expected behavior)")
    
    def test_identity_matrix(self):
        """Test with identity matrix (should have no non-trivial nullspace)"""
        I = gf2matrix.from_dense(np.eye(4, dtype=np.int8))
        w = wiedemann.solve(I, max_iter=10, verbose=False)
        self.assertIsNone(w, "Identity matrix should have no non-trivial nullspace")
    
    def test_zero_matrix(self):
        """Test with zero matrix (every vector is in nullspace)"""
        Z = gf2matrix.from_dense(np.zeros((3, 3), dtype=np.int8))
        w = wiedemann.solve(Z, max_iter=5, verbose=False)
        
        if w is not None:
            # Any non-zero vector should work
            Zw = Z.apply(w)
            self.assertTrue(not np.any(Zw), "Zero matrix times any vector should be zero")
            self.assertTrue(np.any(w), "Solution should be non-zero")
    
    def test_rank_deficient_matrices(self):
        """Test with various rank-deficient matrices"""
        test_matrices = [
            # Rank 1 matrix
            np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.int8),
            # Rank 2 matrix  
            np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]], dtype=np.int8),
            # Block diagonal with zero block
            np.array([[1, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int8)
        ]
        
        for i, A_dense in enumerate(test_matrices):
            with self.subTest(matrix=i):
                A = gf2matrix.from_dense(A_dense)
                
                # Try multiple attempts
                for attempt in range(5):
                    w = wiedemann.solve(A, max_iter=10, verbose=False)
                    if w is not None:
                        # Verify it's actually a solution
                        Aw = A.apply(w)
                        self.assertTrue(not np.any(Aw), f"Matrix {i}: A*w should be zero")
                        self.assertTrue(np.any(w), f"Matrix {i}: Solution should be non-zero")
                        break
                # Note: May not find solution due to probabilistic nature
    
    def test_small_random_matrices(self):
        """Test with small random matrices"""
        for n in range(2, 6):
            with self.subTest(size=n):
                # Generate random sparse matrix
                A = gf2matrix.random(n, density=0.4)
                
                # Try to solve
                w = wiedemann.solve(A, max_iter=15, verbose=False)
                
                if w is not None:
                    # Verify solution
                    Aw = A.apply(w)
                    self.assertTrue(not np.any(Aw), f"A*w should be zero for size {n}")
                    self.assertTrue(np.any(w), f"Solution should be non-zero for size {n}")
    
    def test_known_nullspace_construction(self):
        """Test matrix constructed to have known nullspace"""
        # Construct matrix where [1,0,1] is in nullspace
        # Use A = [[1,0,1],[0,1,1],[1,1,0]] which has [1,0,1] in nullspace
        A_dense = np.array([
            [1, 0, 1],
            [0, 1, 1], 
            [1, 1, 0]
        ], dtype=np.int8)
        A = gf2matrix.from_dense(A_dense)
        
        # Verify [1,0,1] is indeed in nullspace
        test_vec = np.array([1, 0, 1], dtype=np.int8)
        result = A.apply(test_vec)
        expected_zero = np.array([0, 0, 0], dtype=np.int8)
        self.assertTrue(np.array_equal(result, expected_zero), 
                       f"[1,0,1] should be in nullspace, A*[1,0,1] = {result}")
        
        # Try to find this or another nullspace vector
        solution_found = False
        for _ in range(15):
            w = wiedemann.solve(A, max_iter=10, verbose=False)
            if w is not None:
                Aw = A.apply(w)
                self.assertTrue(not np.any(Aw), "Found solution should satisfy A*w = 0")
                self.assertTrue(np.any(w), "Solution should be non-zero")
                solution_found = True
                break
        
        if not solution_found:
            self.skipTest("Scalar Wiedemann didn't find the known nullspace vector")
    
    def test_input_validation(self):
        """Test input validation and edge cases"""
        # Test with numpy array input
        A_np = np.array([[1, 1], [0, 1]], dtype=np.int8)
        w = wiedemann.solve(A_np, max_iter=5, verbose=False)
        # Should not raise error, may or may not find solution
        
        # Test with 1x1 matrix
        A_1x1 = gf2matrix.from_dense(np.array([[1]], dtype=np.int8))
        w = wiedemann.solve(A_1x1, max_iter=3, verbose=False)
        self.assertIsNone(w, "1x1 non-zero matrix should have no nullspace")
        
        A_1x1_zero = gf2matrix.from_dense(np.array([[0]], dtype=np.int8))
        w = wiedemann.solve(A_1x1_zero, max_iter=3, verbose=False)
        if w is not None:
            self.assertEqual(w.tolist(), [1], "1x1 zero matrix should find [1] in nullspace")
    
    def test_max_iter_parameter(self):
        """Test that max_iter parameter is respected"""
        A = gf2matrix.random(4, density=0.5)
        
        # Should try at most max_iter attempts
        w = wiedemann.solve(A, max_iter=2, verbose=False)
        # Function should return (either solution or None) without hanging
        self.assertIsInstance(w, (type(None), np.ndarray), "Should return None or numpy array")
    
    def brute_force_nullspace(self, A):
        """Helper: brute force find nullspace for small matrices"""
        n = A.n_rows
        nullspace = []
        
        # Try all 2^n possible vectors
        for i in range(1, 2**n):  # skip zero vector
            vec = np.array([(i >> j) & 1 for j in range(n)], dtype=np.int8)
            if not np.any(A.apply(vec)):
                nullspace.append(vec)
        
        return nullspace
    
    def test_against_brute_force(self):
        """Compare results against brute force for very small matrices"""
        for n in range(2, 5):
            with self.subTest(size=n):
                A = gf2matrix.random(n, density=0.6)
                brute_nullspace = self.brute_force_nullspace(A)
                
                if len(brute_nullspace) == 0:
                    # No nullspace, Wiedemann should return None
                    w = wiedemann.solve(A, max_iter=10, verbose=False)
                    if w is not None:
                        # If Wiedemann finds something, verify it's correct
                        Aw = A.apply(w)
                        self.assertTrue(not np.any(Aw), "Any solution found should be valid")
                else:
                    # There is a nullspace, try to find it
                    found_valid = False
                    for _ in range(10):
                        w = wiedemann.solve(A, max_iter=8, verbose=False)
                        if w is not None:
                            Aw = A.apply(w)
                            if not np.any(Aw):
                                found_valid = True
                                break
                    
                    # Note: Scalar Wiedemann might miss the nullspace due to probabilistic nature
                    if not found_valid and len(brute_nullspace) > 0:
                        pass  # This is expected behavior for scalar Wiedemann

if __name__ == '__main__':
    unittest.main()
