import numpy as np

def LinearTriangulation(K, T1, R1, T2, R2, x1, x2):
    # https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
    I = np.identity(3)
    T1 = np.reshape(T1, (3, 1))
    T2 = np.reshape(T2, (3, 1))
    # projection matrices
    P1 = np.dot(K, np.dot(R1, np.hstack((I, -T1))))
    P2 = np.dot(K, np.dot(R2, np.hstack((I, -T2))))

    p1T = P1[0,:].reshape(1,4)
    p2T = P1[1,:].reshape(1,4)
    p3T = P1[2,:].reshape(1,4)

    p_dash_1T = P2[0,:].reshape(1,4)
    p_dash_2T = P2[1,:].reshape(1,4)
    p_dash_3T = P2[2,:].reshape(1,4)

    X_all = []

    for i in range(x1.shape[0]):
        x = x1[i, 0]
        y = x1[i, 1]
        x_dash = x2[i, 0]
        y_dash = x2[i, 1]

        A = []
        A.append((y * p3T) -  p2T)
        A.append(p1T -  (x * p3T))
        A.append((y_dash * p_dash_3T) -  p_dash_2T)
        A.append(p_dash_1T -  (x_dash * p_dash_3T))
        A = np.array(A).reshape(4,4)

        _, _, vt = np.linalg.svd(A)
        V = vt.T
        X = V[:, -1]
        X_all.append(X)

    return np.array(X_all)