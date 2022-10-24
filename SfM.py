from NonLinearTriangulation import NonLinearTriangulation
from Utils.DataLoader import extractMatchesFromFile
from Utils.ImageUtils import readImageSet, showMatches
from Utils.MiscUtils import ReprojectionError
from GetInlinersRANSAC import processInliners
from GetEssentialMatrix import getEssentialMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguatePose
from PnPRansac import PnPRANSAC
from LinearPnP import reprojectionErrorPnP
from NonLinearPnP import NonLinearPnP
import numpy as np
import os
from tqdm import tqdm
import cv2

folder_name = r"C:\Users\siddh\Documents\733\SfM\Data"
save_path = r"C:\Users\siddh\Documents\733\SfM\outputs"
load_data = False
images = readImageSet(folder_name)
total_images = len(images)
feature_x, feature_y, \
    feature_flag, feature_descriptor = \
        extractMatchesFromFile(folder_name, total_images)

loadF = True
f_matrix = np.empty(shape=(total_images, total_images), dtype=object)
filtered_feature_flag = np.zeros_like(feature_flag)

for i in tqdm(range(0, total_images-1)):
    for j in range(i+1, total_images):
        idx = np.where(feature_flag[:, i] & feature_flag[:, j])
        pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
        pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
        # showMatches(images[i], images[j], pts1, pts2,
        #            fileName=os.path.join(save_path, (str(i)+str(j)+".png")))
        if not loadF:
            idx = np.array(idx).reshape(-1)
            if len(idx) > 7:
                final_f, final_indices = processInliners(pts1, pts2, idx)
                print('At image : ',  i, j, '|| Number of inliers: ', len(final_indices), '/', len(idx))
                f_matrix[i, j] = final_f
                filtered_feature_flag[final_indices, i] = 1
                filtered_feature_flag[final_indices, j] = 1
if not loadF:
    np.save('filtered_feature_flag_.npy',filtered_feature_flag)
    np.save('f_matrix_.npy',f_matrix)
else:
    filtered_feature_flag = np.load('filtered_feature_flag.npy', allow_pickle=True)
    f_matrix = np.load('f_matrix.npy', allow_pickle=True)

# Registration of first two images
print('Registering images 1 and 2')
K = np.array([[568.996140852,             0,  643.21055941],
              [            0, 568.988362396, 477.982801038],
              [            0,             0,             1]]).reshape(3,3)
m, n = 0, 1
F12 = f_matrix[m, n]
E12 = getEssentialMatrix(K, F12)
print('Essential Matrix: \n', E12)

print('Estimating poses of Camera 2')
R_set, T_set = ExtractCameraPose(E12)
idx = np.where(filtered_feature_flag[:, m] & filtered_feature_flag[:, n])
# correspondences
x1 = np.hstack((feature_x[idx, m].reshape((-1, 1)), feature_y[idx, m].reshape((-1, 1))))
x2 = np.hstack((feature_x[idx, n].reshape((-1, 1)), feature_y[idx, n].reshape((-1, 1))))
# reference frame 1st camera
R1 = np.identity(3)
T1 = np.zeros(3)
pts_3D = []
for i in range(len(R_set)):
    X = LinearTriangulation(K, T1, R1, T_set[i], R_set[i], x1, x2)
    X = X/X[:, 3].reshape(-1, 1)
    pts_3D.append(X)

# disambiguate pose out of 4 combinations
R2, T2, X = DisambiguatePose(R_set, T_set, pts_3D)
X = X/X[:, 3].reshape(-1, 1)

# calculate the reprojection error
mean_error1 = ReprojectionError(X, x1, x2, R1, T1, R2, T2, K )
print('Mean error before NonLinearTriangulation: ', mean_error1)

# Non-linear triangulation
print('Performing Non-linear Triangulation')
X_refined = NonLinearTriangulation(K, x1, x2, X, R1, T1, R2, T2)
X_refined = X_refined / X_refined[:,3].reshape(-1,1)
mean_error2 = ReprojectionError(X_refined, x1, x2, R1, T1, R2, T2, K)
print('Mean error after NonLinearTriangulation: ', mean_error2)

# storing all info to register other cameras
X_all = np.zeros((feature_x.shape[0], 3))
camera_indices = np.zeros((feature_x.shape[0], 1), dtype = int) 
X_found = np.zeros((feature_x.shape[0], 1), dtype = int)
X_all[idx] = X[:, :3]
X_found[idx] = 1
camera_indices[idx] = 1
# set X found to zero where the z is negative
X_found[np.where(X_all[:, 2] < 0)] = 0
print('Registered Cameras 1 and 2')

T_set_, R_set_ = [], []
T_set_.append(T1)
R_set_.append(R1)
T_set_.append(T2)
R_set_.append(R2)

print('Registering remaining cameras')
for i in range(2, total_images):
    print('\nRegistering Image: ', str(i+1), ' with reference camera')
    # X 3D from previous registration and i-th camera correspondence
    feature_idx_i = np.where(X_found[:, 0] & filtered_feature_flag[:, i])
    # check if correspondences are less than 8
    if len(feature_idx_i[0]) < 8:
        print('Not enough correspondences, skip')
        continue
    pts_i = np.hstack((feature_x[feature_idx_i, i].reshape(-1,1), feature_y[feature_idx_i, i].reshape(-1,1)))
    # get the actual X 3D correspondences between reference camera and i-th camera
    X = X_all[feature_idx_i, :].reshape(-1,3)
    # perform PnP
    R_init, T_init = PnPRANSAC(K, pts_i, X, n_iterations=1000, error_thresh=5)
    print('Reprojection error = ', reprojectionErrorPnP(X, pts_i, K, R_init, T_init))

    # perform Non-linear PnP
    Ri, Ti = NonLinearPnP(K, pts_i, X, R_init, T_init)
    print('Reprojection error after NonLinear PnP = ', reprojectionErrorPnP(X, pts_i, K, Ri, Ti))
    T_set_.append(Ti)
    R_set_.append(Ri)

    #trianglulation with respect to camera i
    for j in range(0, i):
        idx_X_pts = np.where(filtered_feature_flag[:, j] & filtered_feature_flag[:, i])
        if (len(idx_X_pts[0]) < 8):
            continue

        x1 = np.hstack((feature_x[idx_X_pts, j].reshape((-1, 1)), feature_y[idx_X_pts, j].reshape((-1, 1))))
        x2 = np.hstack((feature_x[idx_X_pts, i].reshape((-1, 1)), feature_y[idx_X_pts, i].reshape((-1, 1))))

        X = LinearTriangulation(K, T_set_[j], R_set_[j], Ti, Ri, x1, x2)
        X = X/X[:,3].reshape(-1,1)
        
        LT_error = ReprojectionError(X, x1, x2, R_set_[j], T_set_[j], Ri, Ti, K)
        
        X = NonLinearTriangulation(K, x1, x2, X, R_set_[j], T_set_[j], Ri, Ti)
        X = X/X[:,3].reshape(-1,1)
        
        nLT_error = ReprojectionError(X, x1, x2, R_set_[j], T_set_[j], Ri, Ti, K)
        print("Error after linear triangulation: ", LT_error, " Error after non linear triangulation: ", nLT_error)

        X_all[idx_X_pts] = X[:,:3]
        X_found[idx_X_pts] = 1
        
        print("appended ", len(idx_X_pts[0]), " points between ", j ," and ", i)

    np.save('optimized_C_set_', T_set_)
    np.save('optimized_R_set_', R_set_)
    np.save('optimized_X_all', X_all)
    np.save('optimized_X_found', X_found)
