from Utils.DataLoader import extractMatchesFromFile
from Utils.ImageUtils import readImageSet

folder_name = r"C:\Users\siddh\Documents\733\SfM\Data"
save_path = r"C:\Users\siddh\Documents\733\SfM\outputs"
load_data = False
images = readImageSet(folder_name)
total_images = len(images)
feature_x, feature_y, \
    feature_flag, feature_descriptor = \
        extractMatchesFromFile(folder_name, total_images)