from Utils.DataLoader import extractMatchesFromFile
from Utils.ImageUtils import readImageSet, showMatches
from GetInlinersRANSAC import processInliners
import numpy as np
import os
from tqdm import tqdm

folder_name = r"C:\Users\siddh\Documents\733\SfM\Data"
save_path = r"C:\Users\siddh\Documents\733\SfM\outputs"
load_data = False
images = readImageSet(folder_name)
total_images = len(images)
feature_x, feature_y, \
    feature_flag, feature_descriptor = \
        extractMatchesFromFile(folder_name, total_images)

f_matrix = np.empty(shape=(total_images, total_images), dtype=object)
filtered_feature_flag = np.zeros_like(feature_flag)

for i in tqdm(range(0, total_images-1)):
    for j in range(i+1, total_images):
        idx = np.where(feature_flag[:, i] & feature_flag[:, j])
        pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
        pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
        # showMatches(images[i], images[j], pts1, pts2,
        #            fileName=os.path.join(save_path, (str(i)+str(j)+".png")))
        idx = np.array(idx).reshape(-1)
        if len(idx) > 7:
            final_f, final_indices = processInliners(pts1, pts2, idx)
            print('At image : ',  i, j, '|| Number of inliers: ', len(final_indices), '/', len(idx))
            f_matrix[i, j] = final_f
            filtered_feature_flag[final_indices, i] = 1
            filtered_feature_flag[final_indices, j] = 1

np.save('filtered_feature_flag_.npy',filtered_feature_flag)
np.save('f_matrix_.npy',f_matrix)
