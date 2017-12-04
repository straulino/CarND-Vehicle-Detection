## Vehicle Detection

Below we cover briefly how we addressed each of the rubric points. The Jupyter notebook provides more detailed insight into the pipeline, as well as a larger array of images illustrating our approach. 

## 1. Extracting HOG features

To calibrate the camera, we are using the 20 chessboard images that we were provided by Udacity. Using the find Chessboard corners of the cv2 library, we identify a 9 by 6 chessboard in each of them, and with these points we can find the distortion transformation of the lens.
Of these 20 images, we noticed that on three of them we can not find the board. Inspecting them (images 1, 4 and 5) we notice that en each of them at least one of the chessboard corners is cropped. If we really wanted to use them, we could change the size of the chessboard to find, but instead they served as an easy way to test that the calibration was indeed working.

<img src="examples/hog.png" width="720" alt="Combined Image" />




