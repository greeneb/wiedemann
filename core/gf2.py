import numpy as np

class gf2:
    """
    Implementation of the Galois Field GF(2) operations.
    
    In GF(2), there are only two elements: 0 and 1.
    Addition corresponds to XOR, and multiplication corresponds to AND.
    
    Operations are done bitwise on integers, treating them as binary representations.
    """
    
    @staticmethod
    def add(a, b):
        """Addition in GF(2) is XOR"""
        return a ^ b
    
    @staticmethod
    def mul(a, b):
        """Multiplication in GF(2) is AND"""
        return a & b
    
    @staticmethod
    def neg(a):
        """Negation in GF(2) is identity (since -1 = 1 in GF(2))"""
        return a
    
    @staticmethod
    def inv(a):
        """
        Multiplicative inverse in GF(2)
        1^(-1) = 1, 0 has no inverse
        """
        if a == 0:
            raise ValueError("Zero has no multiplicative inverse in GF(2)")
        return 1
    
    @staticmethod
    def div(a, b):
        """Division in GF(2)"""
        if b == 0:
            raise ValueError("Division by zero in GF(2)")
        return a  # In GF(2), a/1 = a


class gf2matrix:
    """
    Implementation of a sparse matrix over GF(2).
    
    Matrix entries are stored as a list of (row, col) tuples where non-zero entries are located.
    This is essentially a coordinate (COO) representation optimized for GF(2).
    """
    
    def __init__(self, n, m=None):
        """
        Initialize an n x m sparse matrix over GF(2).
        
        Args:
            n: Number of rows
            m: Number of columns (defaults to n for a square matrix)
        """
        self.n_rows = n
        self.n_cols = m if m is not None else n
        self.entries = set()  # Set of (row, col) tuples for non-zero entries
    
    def set_entry(self, i, j, value):
        """
        Set the entry at position (i, j) to the given value.
        
        Args:
            i: Row index
            j: Column index
            value: Value (0 or 1)
        """
        if i < 0 or i >= self.n_rows or j < 0 or j >= self.n_cols:
            raise IndexError(f"Matrix indices ({i}, {j}) out of bounds")
        
        if value == 1:
            self.entries.add((i, j))
        else:
            self.entries.discard((i, j))
    
    def get_entry(self, i, j):
        """
        Get the entry at position (i, j).
        
        Args:
            i: Row index
            j: Column index
            
        Returns:
            The value at position (i, j) (0 or 1)
        """
        if i < 0 or i >= self.n_rows or j < 0 or j >= self.n_cols:
            raise IndexError(f"Matrix indices ({i}, {j}) out of bounds")
        
        return 1 if (i, j) in self.entries else 0
    
    def apply(self, vector):
        """
        Apply this matrix to a vector (matrix-vector multiplication).
        
        Args:
            vector: A vector of length equal to the number of columns in this matrix
            
        Returns:
            The result of matrix-vector multiplication
        """
        if len(vector) != self.n_cols:
            raise ValueError(f"Vector length ({len(vector)}) does not match matrix columns ({self.n_cols})")
        
        result = np.zeros(self.n_rows, dtype=np.int8)
        
        # For each non-zero entry (i, j) in the matrix
        for i, j in self.entries:
            # If the corresponding vector element is 1, flip the result bit
            if vector[j] == 1:
                result[i] ^= 1
                
        return result
    
    def to_dense(self):
        """
        Convert to a dense numpy array representation.
        
        Returns:
            A dense numpy array representation of this matrix
        """
        matrix = np.zeros((self.n_rows, self.n_cols), dtype=np.int8)
        for i, j in self.entries:
            matrix[i, j] = 1
        return matrix
    
    @classmethod
    def from_dense(cls, matrix):
        """
        Create a sparse matrix from a dense numpy array.
        
        Args:
            matrix: A dense numpy array
            
        Returns:
            A gf2matrix equivalent to the input matrix
        """
        n_rows, n_cols = matrix.shape
        sparse_matrix = cls(n_rows, n_cols)
        for i in range(n_rows):
            for j in range(n_cols):
                if matrix[i, j] == 1:
                    sparse_matrix.set_entry(i, j, 1)
        return sparse_matrix
    
    @classmethod
    def random(cls, n, m=None, density=0.1):
        """
        Create a random sparse matrix with the specified density.
        
        Args:
            n: Number of rows
            m: Number of columns (defaults to n for a square matrix)
            density: Approximate fraction of non-zero entries
            
        Returns:
            A random sparse matrix
        """
        m = m if m is not None else n
        sparse_matrix = cls(n, m)
        
        # Number of non-zero entries
        num_entries = int(n * m * density)
        
        # Generate random positions for non-zero entries
        entries = set()
        # Ensure we get unique entries
        # TODO: specify seed for reproducibility
        while len(entries) < num_entries:
            i = np.random.randint(0, n)
            j = np.random.randint(0, m)
            entries.add((i, j))
        
        # Set the entries
        for i, j in entries:
            sparse_matrix.set_entry(i, j, 1)
        
        return sparse_matrix
    
    def __str__(self):
        """String representation of the matrix"""
        return str(self.to_dense())
    
    def __repr__(self):
        """Detailed string representation of the matrix"""
        return f"gf2matrix({self.n_rows}, {self.n_cols}) with {len(self.entries)} non-zero entries"