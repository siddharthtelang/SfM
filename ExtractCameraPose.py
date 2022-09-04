import numpy as np

def ExtractCameraPose(E):
    # E = tx.R: tx: skew symmetric matrix, R: rotation matrix(orthogonal)
    # tx.T = 0 (T = [x, y, z] translation), so translation is in the nullspace of tx
    U, S, VT = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R, C = [], []
    # four possible combinations
    R.append(np.dot(U, np.dot(W, VT)))
    R.append(np.dot(U, np.dot(W, VT)))
    R.append(np.dot(U, np.dot(W.T, VT)))
    R.append(np.dot(U, np.dot(W.T, VT)))
    C.append(U[:, 2])
    C.append(-U[:, 2])
    C.append(U[:, 2])
    C.append(-U[:, 2])

    # check sign of Rotation matrix, if negative correct by negation
    for i in range(len(R)):
        if (np.linalg.det(R[i]) < 0):
            R[i] = -R[i]
            C[i] = -C[i]
    return R, C