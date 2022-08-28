import os
import numpy as np
import cv2

def extractMatchesFromFile(folder: str, total_images: int):
    feature_descriptor, feature_flag = [], []
    feature_x, feature_y = [], []

    for n in range(1, total_images):
        matching_file_name = os.path.join(folder, "matching" + str(n) + ".txt")
        with open (matching_file_name, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "nFeatures" in line:
                    line = line.split(":")
                    nFeatures = int(line[1])
                else:
                    ux_row = np.zeros((1, total_images)) # all ux matches with corresponding image id as col
                    uy_row = np.zeros((1, total_images)) # all uy matches with corresponding image id as col
                    flag_row = np.zeros((1, total_images), dtype=int) # flag as 1 if there is a match
                    line_elements = line.strip().split()
                    features = [float(x) for x in line_elements]
                    features =np.array(features)

                    n_matches = features[0]
                    r, g, b = features[1], features[2], features[3]
                    feature_descriptor.append([r, g, b])

                    # ux_src, uy_src: ux, uy source image
                    ux_row[0, n-1], uy_row[0, n-1] = features[4], features[5]
                    flag_row[0, n-1] = 1

                    m = 1
                    while n_matches > 1:
                            image_id = int(features[m+5])
                            # next image ux, uy, and set flag
                            ux_row[0, image_id-1], uy_row[0, image_id-1] = features[m+6], features[m+7]
                            flag_row[0, image_id-1] = 1
                            # go to next
                            m += 3
                            # decrease counter
                            n_matches -= 1

                    feature_x.append(ux_row)
                    feature_y.append(uy_row)
                    feature_flag.append(flag_row)

    return np.array(feature_x).reshape(-1, total_images),\
           np.array(feature_y).reshape(-1, total_images),\
           np.array(feature_flag).reshape(-1, total_images),\
           np.array(feature_descriptor).reshape(-1, 3)
