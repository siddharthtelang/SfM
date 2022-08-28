import cv2
import os

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