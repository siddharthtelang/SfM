from Utils.DataLoader import extractMatchesFromFile
from Utils.ImageUtils import readImageSet, showMatches
import numpy as np
import os

folder_name = r"C:\Users\siddh\Documents\733\SfM\Data"
save_path = r"C:\Users\siddh\Documents\733\SfM\outputs"
load_data = False
images = readImageSet(folder_name)
total_images = len(images)
feature_x, feature_y, \
    feature_flag, feature_descriptor = \
        extractMatchesFromFile(folder_name, total_images)

for i in range(0, total_images-1):
    for j in range(i+1, total_images):
        idx = np.where(feature_flag[:, i] & feature_flag[:, j])
        pts1 = np.hstack((feature_x[idx, i].reshape((-1, 1)), feature_y[idx, i].reshape((-1, 1))))
        pts2 = np.hstack((feature_x[idx, j].reshape((-1, 1)), feature_y[idx, j].reshape((-1, 1))))
        showMatches(images[i], images[j], pts1, pts2,
                   fileName=os.path.join(save_path, (str(i)+str(j)+".png")))
