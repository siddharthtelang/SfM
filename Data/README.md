# Data description

The data given are a set of 6 images of building in-front of Levine Hall at UPenn, using a GoPro Hero 3 with fisheye lens distortion corrected. Keypoints matching (SIFT keypoints and descriptors used) data is also provided in the same folder for pairs of images. The data folder contains 5 matching files named matching*.txt where * refers to numbers from 1 to 5. For eg., matching3.txt contains the matching between the third image and the fourth, fifth and sixth images, i.e., I3↔I4, I3↔I5 and I3↔I6 . Therefore, matching6.txt does not exist because it is the matching by itself.

The file format of the matching file is described next. Each matching file is formatted as follows for the i th matching file:

nFeatures: (the number of feature points of the ith image - each following row specifies matches across images given a feature location in the ith image.)

Each Row: (the number of matches for the jth feature) (Red Value) (Green Value) (Blue Value) (ucurrent image) (vcurrent image) (image id) (uimage id image) (vimage id image) (image id) (uimage id image) (v_{image id image})

The images are taken at 1280 × 960 resolution and the camera intrinsic parameters K are given in calibration.txt file. You will program this full pipeline guided by the functions described in following sections.