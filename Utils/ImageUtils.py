import cv2
import os
import numpy as np

def readImageSet(folder_name):
    print("Reading images from ", folder_name)
    images = []
    for root, directory, files in os.walk(folder_name):
        for file in files:
            if file.endswith(".jpg"):
                img_path = os.path.join(root, file)
                images.append(cv2.imread(img_path))
                print(f"Loading image {file}")
    return images

def makeImageSizeSame(images):
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis = 0)
    
    images_resized_list = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized_list.append(image_resized)
    return images_resized_list

def showMatches(image1, image2, pts1, pts2, color=(0,0,255), fileName=""):
    image_1, image_2 = makeImageSizeSame([image1, image2])
    concatenated = np.concatenate((image_1, image_2), axis=1)
    if pts1 is not None:
        pts1_x = pts1[:, 0].astype(int)
        pts1_y = pts1[:, 1].astype(int)
        pts2_x = pts2[:, 0].astype(int) + image_1.shape[1]
        pts2_y = pts2[:, 1].astype(int)
        for i in range(pts1_x.shape[0]):
            cv2.line(concatenated, (pts1_x[i], pts1_y[i]), (pts2_x[i], pts2_y[i]), color, 1)
        cv2.imshow(fileName, concatenated)
        cv2.imwrite(fileName, concatenated)
        cv2.waitKey()
        cv2.destroyAllWindows()
