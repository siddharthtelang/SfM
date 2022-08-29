import numpy as np

def normalize(uv: np.ndarray):
    """https://www.cc.gatech.edu/classes/AY2016/cs4476_fall/results/proj3/html/sdai30/index.html

    Args:
        uv (np.ndarray): feature points
    """
    # calculate the mean of all the points and subtract the mean: recenter the points
    uv_ = np.mean(uv, axis=0)
    u_, v_ = uv_[0], uv_[1]
    u_cap, v_cap = uv[:, 0] - u_, uv[:, 1] - v_
    # scale term s to be the average distances of the centered points
    # from the origin
    s = (2/np.mean(u_cap**2 + v_cap**2))**(0.5)
    # Rotate by s and translate by the mean centroid
    T_scale = np.diag([s,s,1])
    T_translation = np.array([[1, 0, -u_],[0, 1, -v_],[0, 0 , 1]])
    T = np.dot(T_scale, T_translation)
    x_ = np.column_stack((uv, np.ones(len(uv))))
    x_norm = (T.dot(x_.T)).T

    return x_norm, T

def EstimateFundamentalMatrix(pts1, pts2):
    # 8-point algorithm
    if pts1.shape[0] > 7:
        x1_norm, T1 = normalize(pts1)
        x2_norm, T2 = normalize(pts2)
        A = np.zeros((len(x1_norm),9))
        for i in range(0, len(x1_norm)):
            x_1, y_1 = x1_norm[i][0], x1_norm[i][1]
            x_2,y_2 = x2_norm[i][0], x2_norm[i][1]
            A[i] = np.array([x_1*x_2, x_2*y_1, x_2, y_2*x_1, y_2*y_1, y_2, x_1, y_1, 1])

        U, S, VT = np.linalg.svd(A, full_matrices=True)
        # take last column of V / row of VT corresponding to the least singular value
        F = VT.T[:, -1]
        F = F.reshape(3,3)

        # decompose, enforce rank of 2
        u, s, vt = np.linalg.svd(F)
        s = np.diag(s)
        s[2,2] = 0
        F = np.dot(u, np.dot(s, vt))

        # de-normalize
        F = np.dot(T2.T, np.dot(F, T1))
        F = F / F[2,2]
        return F
    else:
        return None