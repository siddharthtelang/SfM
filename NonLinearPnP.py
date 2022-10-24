from Utils.MiscUtils import getQuaternion, getRotation, ProjectionMatrix
import numpy as np
import scipy.optimize as optimize

def makeHomogenous(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))

def NonLinearPnP(K, pts, x3D, R0, C0):
    """    
    K : Camera Matrix
    pts1, pts2 : Point Correspondences
    x3D :  initial 3D point 
    R2, C2 : relative camera pose - estimated from PnP
    Returns:
        x3D : optimized 3D points
    """

    Q = getQuaternion(R0)
    X0 = [Q[0] ,Q[1], Q[2], Q[3], C0[0], C0[1], C0[2]]

    optimized_params = optimize.least_squares(
        fun = PnPLoss,
        x0=X0,
        method="trf",
        args=[x3D, pts, K])
    X1 = optimized_params.x
    Q = X1[:4]
    C = X1[4:]
    R = getRotation(Q)
    return R, C

def PnPLoss(X0, x3D, pts, K):
    Q, C = X0[:4], X0[4:].reshape(-1,1)
    R = getRotation(Q)
    P = ProjectionMatrix(R, C, K)
    
    X = makeHomogenous(x3D)
    XT = X.T
    uv1 = np.dot(P, XT)
    uv1_ = uv1 / uv1[2, :]
    uv1_ = uv1_.T
    uv1_ = uv1_[:, 0:2]
    p1 = uv1_
    e1 = np.sum(np.square(np.linalg.norm(pts-p1, axis=0)))/X.shape[0]
    return e1