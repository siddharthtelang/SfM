import numpy as np

def DepthPositiveConstraint(x3d, r3, T):
    n_positive_z = 0
    for x in x3d:
        x = x.reshape(-1, 1)
        if r3.dot(x-T) > 0 and x[2] > 0:
            n_positive_z += 1
    return n_positive_z

def DisambiguatePose(R_set, T_set, X_3D):
    # z: r3(X-C) > 0
    best_i = 0
    max_positive_depth = -np.inf
    for i in range(len(R_set)):
        R, T = R_set[i], T_set[i].reshape(-1, 1)
        r3 = R[2, :].reshape(1, -1)
        x3d = X_3D[i]
        x3d = x3d / x3d[:, 3].reshape(-1, 1)
        x3d = x3d[:, 0:3]
        n_positive_depth = DepthPositiveConstraint(x3d, r3, T)
        print('Positive points at i = ', i, ' = ', n_positive_depth)
        if n_positive_depth > max_positive_depth:
            max_positive_depth = n_positive_depth
            best_i = i
    print('Best i: ', best_i)
    R, T, x3d = R_set[best_i], T_set[best_i], X_3D[best_i]
    return R, T, x3d
