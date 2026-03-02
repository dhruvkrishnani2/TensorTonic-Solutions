import numpy as np

def matrix_transpose(A):
    """
    Return the transpose of matrix A (swap rows and columns).
    """
    # Write code here
    matrix = np.asarray(A)
    m,n = matrix.shape

    result = np.zeros((n,m))   
    
    for i in range(m):
        for j in range(n):
            result[j][i] = matrix[i][j]
            
    return result