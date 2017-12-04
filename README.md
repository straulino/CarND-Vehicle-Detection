## Vehicle Detection

Below we cover briefly how we addressed each of the rubric points. The Jupyter notebook provides more detailed insight into the pipeline, as well as a larger array of images illustrating our approach. 

## 1. Extracting HOG features and training the Classifier

We used skimage's HOG function to extract the HOG features of our images. We also tried using Open CV, but although it is much faster on an individual basis, since we could not easily adapt it to allow for HOG subsampling (it has no functionality to allow it to return a multidimensional array instead of a vector of features), we chose to implement our extractor following the code of the classes.

We then chose to fit a SVM to our training set. Since the Udacity images are sequential and thus can be very similar, we decided to construct the test set purely from the KITTI images.
After testing the accuracy of the classifier trying different colour spaces, orientations, and cells per block, we settled on the YCrCb colour space, and used 8 pixels per cell, 2 cells per block and 9 orientations. Since we achieved 99.36% accuracy, we decided that our classifier should be good enough for the current task, and proceeded to the other parts of the pipeline.

<img src="examples/training.png" width="720" />
<img src="examples/hog.png" width="720"/>


## 2. Window Searching

Once we had trained the classifier, it was time to find the right way of using it to find cars in the video frames. The first thing we did was to limit the region of the images where we would be interested in seatching for cars. So we cropped roughly the top half of the image and the very bottom, where we see the dashboard. 

We then implemented hog sub sampling with a sliding window search. That is, we choose a region and we draw all the windows determined by our parameters (window size and overlap). We extract the HOG features for the entire region, but keep it as a multidimensional array, thus allowing us to subsample the cells of each window without repeating the extraction. Whenever the classifier identifies a car, we keep that window. Below is an image showcasing the output of this approach.

<img src="examples/window.png" width="720" alt="Combined Image" />

Notice that there might be multiple windows for any given car. We also found a few false positives in our first implementation, but we were able to severly reduce them by raising the classifier threshold. Instead of using 50% probability as the cutoff between car and no car, we made it 60%. This was enough to see a significant improvement.

We then played around with different sliding window parameters. It should be obvious that the apparent size of a car in the video is strongly correlated with the position in the image, and so cars that appear closer will also look bigger. Using this insight, we ended up with 4 similar regions where we looked for cars ranging in scale from 1 to 2.5, plus two smaller regions to the sides where we looked for cars 3 times larger than our template case.

To make sense of the fact that each car might be identified by more than one window, and also to leverage past information in the video (ie, cars do not appear and dissappear frame on frame), we used a heatmap. Each time a pixel is part of a window that was classified as a car, we increase its cound by 1 (starting from zero of course). We then use a threshold to discard pixels that appear in only few of the windows. This way we are more likely to deal correctly with false positives, and we have a more coherent location for the cars. By doing this not only by frame, but also keeping the information over the last 25 frames, we were able to better capture the cars' position and somewhat smooth the shape of the markers that track them.


<img src="examples/heatmap.png" width="720" alt="Combined Image" />
