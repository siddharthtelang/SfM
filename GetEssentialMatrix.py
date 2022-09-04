import numpy as np

def getEssentialMatrix(K, F):
    """Get essential matrix from Fundamental Matrix and Camera intrinsic

    Args:
        K (np.ndarray): Intrinsic matrix
        F (np.ndarray): Fundamental matrix
    """
    E = np.dot(K.T, np.dot(F, K))
    # enforce rank 2
    U, S, VT = np.linalg.svd(E)
    S = np.diag([1, 1, 0])
    E = np.dot(U, np.dot(S, VT))
    return E