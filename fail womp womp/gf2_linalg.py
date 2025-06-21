import numpy as np

class GF2Vector:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.uint8) & 1

    def __add__(self, other):
        return GF2Vector(self.data ^ other.data)

    def dot(self, other):
        return int(np.bitwise_and(self.data, other.data).sum() & 1)

    def __repr__(self):
        return f"GF2Vector({self.data.tolist()})"

class GF2Matrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=np.uint8) & 1

    def __add__(self, other):
        return GF2Matrix(self.data ^ other.data)

    def __matmul__(self, other):
        if isinstance(other, GF2Vector):
            # Matrix-vector multiplication over GF(2)
            result = np.bitwise_and(self.data, other.data).sum(axis=1) & 1
            return GF2Vector(result)
        elif isinstance(other, GF2Matrix):
            # Matrix-matrix multiplication over GF(2)
            result = np.zeros((self.data.shape[0], other.data.shape[1]), dtype=np.uint8)
            for i in range(self.data.shape[0]):
                for j in range(other.data.shape[1]):
                    result[i, j] = np.bitwise_and(self.data[i], other.data[:, j]).sum() & 1
            return GF2Matrix(result)
        else:
            raise TypeError("Unsupported operand type(s)")

    def transpose(self):
        return GF2Matrix(self.data.T)

    def row_echelon(self):
        A = self.data.copy()
        m, n = A.shape
        r = 0
        for c in range(n):
            pivot = np.where(A[r:, c] == 1)[0]
            if pivot.size == 0:
                continue
            i = pivot[0] + r
            if i != r:
                A[[r, i]] = A[[i, r]]
            for j in range(r+1, m):
                if A[j, c] == 1:
                    A[j] ^= A[r]
            r += 1
            if r == m:
                break
        return GF2Matrix(A)

    def __repr__(self):
        return f"GF2Matrix({self.data.tolist()})"