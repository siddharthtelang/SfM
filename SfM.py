from Utils.DataLoader import extractMatchesFromFile
from Utils.ImageUtils import readImageSet, showMatches
from GetInlinersRANSAC import processInliners
from GetEssentialMatrix import getEssentialMatrix
from ExtractCameraPose import ExtractCameraPose
from LinearTriangulation import LinearTriangulation
from DisambiguateCameraPose import DisambiguatePose
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

