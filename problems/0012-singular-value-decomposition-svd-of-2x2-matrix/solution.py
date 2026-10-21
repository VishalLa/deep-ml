import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    
    a, b, c, d = A[0, 0], A[0, 1], A[1, 0], A[1, 1]
    
    # Step 1: right singular vectors from one Jacobi rotation
    # M = A^T A is symmetric with entries [[p, q], [q, r]]
    M = A.T @ A
    p, q, r = M[0, 0], M[0, 1], M[1, 1]
    
    theta = 0.5 * np.arctan2(2 * q, p - r)   # tan(2θ) = 2q / (p - r)
    cs, sn = np.cos(theta), np.sin(theta)
    V = np.array(
        [[cs, -sn],
        [sn,  cs]]
    )

    # The diagonal of V^T M V holds the eigenvalues σ1², σ2²; sort by them
    lam = np.diag(V.T @ M @ V)
    order = np.argsort(-lam)
    V = V[:, order]
    
    # Step 2: singular values from the closed form
    T = a**2 + b**2 + c**2 + d**2       # trace of A^T A
    Delta = abs(a * d - b * c)          # |det A| = σ1 σ2
    
    S1 = (np.sqrt(T + 2 * Delta) + np.sqrt(max(T - 2 * Delta, 0.0))) / 2
    # σ2 = |det A| / σ1 avoids cancellation in the subtraction form
    S2 = Delta / S1 if S1 > 0 else 0.0
    S = np.array([S1, S2])
    
    # Step 3: left singular vectors u_i = A v_i / σ_i
    tol = 1e-12 * max(S[0], 1.0)
    U = np.zeros((2, 2))
    
    if S[0] <= tol:
        # A is (numerically) the zero matrix, so any orthogonal U works
        U = np.eye(2)
    else:
        U[:, 0] = A @ V[:, 0] / S[0]
        if S[1] > tol:
            U[:, 1] = A @ V[:, 1] / S[1]
        else:
            # Rank 1: complete U with a unit vector orthogonal to u1
            U[:, 1] = np.array([-U[1, 0], U[0, 0]])

    return U, S, V.T
