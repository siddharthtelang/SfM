import numpy as np
from Utils.MiscUtils import makeHomogenous, ProjectionMatrix

def reprojectionErrorPnP(X, pt1, K, R, C):
    
    P1 = ProjectionMatrix(R, C, K)
    X = makeHomogenous(X)
    XT = X.T
    uv1 = np.dot(P1, XT)
    uv1_ = uv1 / uv1[2, :]
    uv1_ = uv1_.T
    uv1_ = uv1_[:, 0:2]
    p1 = uv1_
    e1 = np.sum(np.square(np.linalg.norm(pt1-p1, axis=0)))
    return e1/X.shape[0]

def LinearPnP(X_set, x_set, K):
    n = X_set.shape[0]
    X4 = makeHomogenous(X_set)
    x3 = makeHomogenous(x_set)
    # projection matrix
    xn = np.linalg.inv(K).dot(x3.T)
    xn = xn.T

    for i in range(n):
        X = X4[i].reshape((1, 4))
        zeros = np.zeros((1, 4))
        u, v, _ = xn[i]
        # skew symmetric
        u_cross = np.array([[0, -1, v],
                            [1, 0, -u],
                            [-v, u, 0]])
        X_tilda = np.vstack((np.hstack((X, zeros, zeros)),
                            np.hstack((zeros, X, zeros)),
                            np.hstack((zeros, zeros, X))))
        # cross product as dot product of skew symmetric with orthogonal
        a = u_cross.dot(X_tilda)
        if i == 0:
            A = a
        else:
            A = np.vstack((A, a))
    # perform SVD
    U, S, VT = np.linalg.svd(A)
    # last row of VT is the nullspace: P
    P = VT[-1].reshape((3, 4))
    # extract the rotation
    R = P[:, :3]
    # decompose and enforce orthogonality
    U, S, VT = np.linalg.svd(R)
    R = U.dot(VT)
    # camera center
    C = P[:, 3]
    # translation
    t = -np.dot(R.T, C)
    t_ = -np.dot(np.linalg.inv(R), C)
    assert t.all() == t_.all()

    # correct the sign
    if np.linalg.det(R) < 0:
        R = -R
        t = -t
    return R, t