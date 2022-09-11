import numpy as np
from scipy.spatial.transform import Rotation 

def ProjectionMatrix(R, C, K):
    C = np.reshape(C, (3, 1))        
    I = np.identity(3)
    P = np.dot(K, np.dot(R, np.hstack((I, -C))))
    return P

def ReprojectionError(X, pt1, pt2, R1, C1, R2, C2, K):
    
    P1 = ProjectionMatrix(R1, C1, K) 
    P2 = ProjectionMatrix(R2, C2, K)
    XT = X.T
    uv1 = np.dot(P1, XT)
    uv1_ = uv1 / uv1[2, :]
    uv1_ = uv1_.T
    uv1_ = uv1_[:, 0:2]
    p1 = uv1_

    uv2 = np.dot(P2, XT)
    uv2_ = uv2 / uv2[2, :]
    uv2_ = uv2_.T
    uv2_ = uv2_[:, 0:2]
    p2 = uv2_

    e1 = np.sum(np.square(np.linalg.norm(pt1-p1, axis=0)))
    e2 = np.sum(np.square(np.linalg.norm(pt2-p2, axis=0)))

    return (e1+e2)/X.shape[0]

def getQuaternion(R2):
    Q = Rotation.from_matrix(R2)
    return Q.as_quat()

def getEuler(R2):
    euler = Rotation.from_matrix(R2)
    return euler.as_rotvec()

def getRotation(Q, type_ = 'q'):
    if type_ == 'q':
        R = Rotation.from_quat(Q)
        return R.as_matrix()
    elif type_ == 'e':
        R = Rotation.from_rotvec(Q)
        return R.as_matrix()

def makeHomogenous(pts):
    return np.hstack((pts, np.ones((pts.shape[0], 1))))