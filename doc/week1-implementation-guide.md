# Week 1 Implementation Guide: Scalar Wiedemann & Berlekamp-Massey

## Overview
Week 1 focuses on implementing the foundational components of the Block Wiedemann algorithm:
- GF(2) field operations
- Berlekamp-Massey algorithm for minimal polynomial finding
- Scalar Wiedemann algorithm for solving sparse linear systems
- Sparse matrix operations over GF(2)

## 1. GF(2) Field Operations

GF(2) is the simplest finite field with only two elements: {0, 1}.

### Basic Operations:
- **Addition**: XOR operation (a ⊕ b)
- **Multiplication**: AND operation (a ∧ b)
- **Negation**: Identity (-a = a, since -1 = 1 in GF(2))
- **Division**: Since only 1 has an inverse (1⁻¹ = 1)

```python
def gf2_add(a, b):
    return a ^ b

def gf2_mul(a, b):
    return a & b
```

## 2. Berlekamp-Massey Algorithm

The Berlekamp-Massey algorithm finds the shortest Linear Feedback Shift Register (LFSR) that can generate a given sequence. In the context of the Wiedemann algorithm, it finds the minimal polynomial of a sequence.

### Key Concepts:
- **Connection Polynomial**: Represents the LFSR coefficients
- **Discrepancy**: Difference between predicted and actual sequence values
- **Linear Complexity**: Degree of the minimal polynomial

### Algorithm Steps:
1. Initialize connection polynomial C(x) = 1
2. For each element in the sequence:
   - Calculate discrepancy d
   - If d ≠ 0, update the connection polynomial
   - Adjust polynomial length if necessary

## 3. Scalar Wiedemann Algorithm

The scalar Wiedemann algorithm solves sparse linear systems Ax = b over finite fields using Krylov sequences and minimal polynomials.

### Algorithm Overview:
1. **Krylov Sequence Generation**: Create sequence {u^T A^i b} for i = 0, 1, ..., 2n-1
2. **Minimal Polynomial**: Use Berlekamp-Massey to find the minimal polynomial
3. **Solution Construction**: Build solution from polynomial coefficients

### Key Formula:
If minimal polynomial is f(x) = c₀ + c₁x + ... + cₑxᵈ, then:
x = (1/c₀) * (c₁A^(d-1)b + c₂A^(d-2)b + ... + cₑb)

## 4. Sparse Matrix Operations

Efficient sparse matrix representation is crucial for performance:

### Coordinate (COO) Format:
Store only non-zero entries as (row, col) pairs. For GF(2), we only need to track positions where entries equal 1.

### Matrix-Vector Multiplication:
For each non-zero entry (i, j) in matrix, if vector[j] = 1, then result[i] ⊕= 1.

## 5. Implementation Structure

```
week1_implementation/
├── gf2_field.py          # GF(2) field operations
├── sparse_matrix.py      # Sparse matrix class
├── berlekamp_massey.py   # Berlekamp-Massey algorithm
├── scalar_wiedemann.py   # Scalar Wiedemann solver
├── tests/                # Unit tests
│   ├── test_gf2.py
│   ├── test_berlekamp.py
│   └── test_wiedemann.py
└── examples/             # Usage examples
    └── solve_example.py
```

## 6. Testing Strategy

### Test Cases:
1. **GF(2) Operations**: Verify XOR/AND behavior
2. **Berlekamp-Massey**: Test with known LFSR sequences
3. **Scalar Wiedemann**: Test with small matrices and known solutions
4. **Sparse Matrices**: Test matrix-vector multiplication

### Validation:
- Verify solutions by computing Ax and comparing with b
- Test with various matrix sizes and sparsity patterns
- Handle edge cases (singular matrices, no solutions)

## 7. Common Pitfalls

1. **Random Vector Selection**: The scalar Wiedemann algorithm is probabilistic; multiple attempts with different random vectors may be needed.

2. **Minimal Polynomial Construction**: Ensure the Berlekamp-Massey implementation correctly handles the GF(2) arithmetic.

3. **Solution Reconstruction**: The formula for building the solution from the minimal polynomial coefficients must be implemented carefully.

4. **Matrix Singularity**: Not all matrices have unique solutions; handle cases where the system is singular or inconsistent.

## 8. Performance Considerations

- **Sparse Storage**: Use efficient data structures for sparse matrices
- **Memory Usage**: Krylov sequences can be large; consider streaming approaches
- **Iteration Limits**: Set reasonable bounds on the number of iterations

## Next Steps (Week 2)
- Extend to block Krylov sequences
- Implement matrix polynomial operations
- Prepare for Block Berlekamp-Massey algorithm