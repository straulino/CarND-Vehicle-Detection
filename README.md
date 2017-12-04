## Vehicle Detection

Below we cover briefly how we addressed each of the rubric points. The Jupyter notebook provides more detailed insight into the pipeline, as well as a larger array of images illustrating our approach. 

## 1. Extracting HOG features and training the Classifier

We used skimage's HOG function to extract the HOG features of our images. We also tried using Open CV, but although it is much faster on an individual basis, since we could not easily adapt it to allow for HOG subsampling (it has no functionality to allow it to return a multidimensional array instead of a vector of features), we chose to implement our extractor following the code of the classes.

We then chose to fit a SVM to our training set. Since the Udacity images are sequential and thus can be very similar, we decided to construct the test set purely from the KITTI images.
After testing the accuracy of the classifier trying different colour spaces, orientations, and cells per block, we settled on the YCrCb colour space, and used 8 pixels per cell, 2 cells per block and 9 orientations. Since we achieved 99.36% accuracy, we decided that our classifier should be good enough for the current task, and proceeded to the other parts of the pipeline.

<img src="examples/train.png" width="720" />
<img src="examples/hog.png" width="720"/>


## 2. Window Searching

Once we had trained the classifier, it was time to find the right way of using it to find cars in the video frames. The first thing we did was to limit the region of the images where we would be interested in seatching for cars. So we cropped roughly the top half of the image and the very bottom, where we see the dashboard. 

We then implemented hog sub sampling with a sliding window search. That is, we choose a region and we draw all the windows determined by our parameters (window size and overlap). We extract the HOG features for the entire region, but keep it as a multidimensional array, thus allowing us to subsample the cells of each window without repeating the extraction. Whenever the classifier identifies a car, we keep that window. Below is an image showcasing the output of this approach.

<img src="window/hog.png" width="720" alt="Combined Image" />
