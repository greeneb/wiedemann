# Wiedemann Algorithm Implementation

A Python implementation of the Wiedemann and Berlekamp-Massey algorithms for solving linear systems over finite fields, with applications to quantum error correction.

## Overview

This project provides implementations of:

1. **Scalar Wiedemann Algorithm** - For finding kernel vectors of matrices over finite fields
2. **Block Wiedemann Algorithm** - An efficient block version for large matrices
3. **Berlekamp-Massey Algorithm** - For finding minimal polynomials of sequences
4. **Block Berlekamp-Massey Algorithm** - Block version for matrix sequences

These algorithms are particularly useful in quantum error correction and computational linear algebra.

## Installation

```bash
pip install -e .
```

## Usage

### Scalar Wiedemann Algorithm

```python
import galois
from wiedemann.scalar_wiedemann import solve

# Create a singular matrix over GF(7)
GF = galois.GF(7)
A = GF([[1, 2, 3],
        [2, 4, 6],
        [1, 1, 1]])

# Find a kernel vector
kernel_vector = solve(A, GF)
print(f"Kernel vector: {kernel_vector}")
print(f"A @ kernel_vector = {A @ kernel_vector}")  # Should be zero
```

### Block Wiedemann Algorithm

```python
import galois
from wiedemann.block_wiedemann import block_wiedemann

# Create a singular matrix
GF = galois.GF(5)
A = GF([[1, 2, 3],
        [2, 4, 1],
        [1, 1, 1]])

# Find a kernel vector using block Wiedemann
kernel_vector = block_wiedemann(A, GF, m=2, n=2)
print(f"Kernel vector: {kernel_vector}")
print(f"A @ kernel_vector = {A @ kernel_vector}")  # Should be zero
```

### Berlekamp-Massey Algorithm

```python
import galois
from wiedemann.scalar_bm import berlekamp_massey

# Create a sequence over GF(5)
GF = galois.GF(5)
sequence = [1, 2, 4, 3, 1, 2, 4, 3]  # Periodic sequence

# Find minimal polynomial
poly = berlekamp_massey(sequence, GF)
print(f"Minimal polynomial: {poly}")
```

## Algorithm Details

### Scalar Wiedemann Algorithm

The scalar Wiedemann algorithm works by:

1. **Sequence Generation**: Generate a scalar sequence `s_t = y^T A^t x` where `x` and `y` are random vectors
2. **Minimal Polynomial**: Use Berlekamp-Massey to find the minimal polynomial of the sequence
3. **Kernel Construction**: Use the minimal polynomial to construct a kernel vector

### Block Wiedemann Algorithm

The block Wiedemann algorithm (Kaltofen 1993) is more efficient for large matrices:

1. **Block Sequence Generation**: Generate a sequence of matrices `S_t = X^T A^t Y`
2. **Block Toeplitz System**: Solve a block Toeplitz system to find recurrence coefficients
3. **Solution Reconstruction**: Use the coefficients to reconstruct a kernel vector

### Berlekamp-Massey Algorithm

The Berlekamp-Massey algorithm finds the shortest linear feedback shift register (LFSR) that generates a given sequence:

1. **Initialization**: Start with constant polynomial
2. **Discrepancy Computation**: Compute discrepancy between predicted and actual sequence values
3. **Polynomial Update**: Update the LFSR polynomial when discrepancy is non-zero
4. **Length Update**: Update the LFSR length when necessary

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

The test suite includes:
- Basic functionality tests
- Edge cases (zero matrices, large matrices)
- Different field sizes
- Different block sizes for block algorithms

## Performance

- **Scalar Wiedemann**: Good for small to medium matrices
- **Block Wiedemann**: More efficient for large matrices, especially sparse ones
- **Fallback Strategy**: Block Wiedemann falls back to scalar Wiedemann if needed

## Applications

This implementation is designed for:
- Quantum error correction research
- Computational linear algebra
- Cryptanalysis
- Sparse linear system solving

## References

- Wiedemann, D. (1986). Solving sparse linear equations over finite fields.
- Kaltofen, E. (1993). Analysis of Coppersmith's block Wiedemann algorithm.
- Berlekamp, E. R. (1968). Algebraic coding theory.
- Massey, J. L. (1969). Shift-register synthesis and BCH decoding.

## License

This project is licensed under the MIT License.
