import numpy as np

def determinant_4x4(
    matrix: list[list[int|float]]
) -> float:
    # Your recursive implementation here
    matrix = np.array(matrix)
    
    def det(mat: np.ndarray) -> int|float:
        n = mat.shape[0]
        if n == 1: return mat[0, 0]
        # if n == 2: return mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
        
        total = 0.0
        
        for j in range(n):
            minor = np.delete(np.delete(mat, 0, axis=0), j, axis=1)
            total += ((-1) ** j) * mat[0, j] * det(minor)
            
        return total
    
    return det(mat=matrix)
