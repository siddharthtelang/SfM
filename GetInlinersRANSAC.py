import numpy as np
from EstimateFundamentalMatrix import EstimateFundamentalMatrix

def calcError(pts1, pts2, F):
    """error = x'^T.F.x, should ideally be zero as per epipolar constraint

    Args:
        pts1 (np.ndarray): points from 1st image
        pts2 (np.ndarray): points from 2nd image
        F (np.ndarray): F matrix
    """
    pts1 = np.hstack((pts1, np.ones(1)))
    pts2 = np.hstack((pts2, np.ones(1)))
    error = np.dot(pts2.T, np.dot(F, pts1))
    return abs(error)

def processInliners(pts1, pts2, idx):
    n_iterations = 2000
    error_threshold = 0.002
    inliners_best = 0
    final_indices = []
    final_f = None

    for i in range(0, n_iterations):
        # randomly select 8 rows
        n_rows = pts1.shape[0]
        random_indices = np.random.choice(n_rows, size=8)
        pts1_8 = pts1[random_indices, :]
        pts2_8 = pts2[random_indices, :]
        f_8 = EstimateFundamentalMatrix(pts1_8, pts2_8)
        indices = []
        if f_8 is not None:
            for j in range(n_rows):
                error = calcError(pts1[j, :], pts2[j, :], f_8)
                if error < error_threshold:
                    indices.append(idx[j])
        if len(indices) > inliners_best:
            inliners_best = len(indices)
            final_indices = indices
            final_f = f_8
    return final_f, final_indices