##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./Results/Car_NonCar.png
[image2]: ./Results/hog_image.jpg
[image3]: ./Results/sliding_windows_medium.png
[image4]: ./Results/sliding_windows_small.png
[image5]: ./Results/final_detection_multiboxes.png
[image6]: ./Results/final_detection_multiboxes_1.png
[image7]: ./Results/combined_heat_map.png
[image8]: ./Results/final_detection_bad_heat_label.png
[image9]: ./Results/final_detection_heat_label.png
[image10]: ./Results/final_detection_heat_label_1.png
[image11]: ./Results/final_detection_boxes.png
[video1]: ./Results/object_detection_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`. An example of the RGB channel histograms are also included below:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

The orientations I tried were 9 and 11 and the two did not seem to exhibit much difference, however the pixels_per_cell parameter effected the output much more. I originally tried 8 (8x8 pixels) and this, along with 3 32-bin histograms for the colorspace, resulted in 8460 features. The Linear SVM took a little longer to train with this many features, but when I increased the pixels_per_cell to 16, the features were greatly reduced. The 4000 or so features allowed the Linear SVM to train faster, however, it didn't effect the speed of video processing, as predictions don't take much time at all (it's mostly the training). Furthermore, the reduced number of features caused the SVM to misclassify a lot of windows and caused the heatmap to be extremely sparse. In the end, I decided that the small performance gain with the reduced feature list was not worth the reduction in accuracy. I kept the cells_per_block at 2 because according to the documentation, it is not necessarily required, yet I believe that if it got too large, the HOG features for each window would be very uniform and windows with lots of variation in HOG features would appear to have less distinct orientations.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in line () of the project.py file. I trained on both the GTI and KITTI images ultimately creating a data set of over 8000 non-car and 8000 car images. The 8640 features I used were composed of a 32 bin histogram for each of the 3 RGB color channels and 9 orientations of 8x8 pixels_per_cell HOG features. The SVM took () seconds to train and the accuracy was about 98.4% each run. I attempted to perform a grid search to optimize the hyperparameters for the LinearSVM but ultimately found it unnecessary. I did discover that while using SVM and trying the rbf kernel that the training was extremely slow and the predictions were unacceptably slow for even a single image. I tried the default 1 and tried 10 for C, the regularization parameter. It did not effect the training speed, validation accuracy nor the test accuracy so I decided to just go with the default 1.   

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided on a 75% overlap in the sliding window search. Also based on the sample images, I estimated that a 1.25 scale along with the original 1.5 scale would help classify cars that are futher away. I attemped to include a 1.75 scale and 2.0 scale window but they created a lot of noise without noticeable gain in accuracy when compiling the heatmaps. Furthermore, the more window scales I added, the slower the pipline became. With the 1.25 scale window, the sliding window search had a lot more windows to classify but at such a small scale, only images in the 400 - 528 y-axis pixel range could yield a car-like image. If the windows were to search lower in the image, the smaller windows would hit only small portions of larger scale cars (just a doors or just a wheel) and not classify it as a car. So this decreased search range for the 1.25 window scale still had acceptable processing time. The 1.5 scale window was used across 400 - 656 y-axis pixel range. I figured that a larger scale window would be needed because cars freshly entering the frame form the adjacent lane would appear large and I did not believe that a 1.5 scale window which only captures about 96x96 pixels would similarly only see a car window or smooth door. However, the 1.5 scale performed well enough and any additional windows would increase frame processing time. Below I am displaying the range of windows I searched. The first image is the 1.5 scale window, and the image below is the 1.25 scale window with half of the search height:

![alt text][image3]
![alt text][image4]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using RGB 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.  Here are some example images where the windows are overlapping on what is classified as a part of a car:

![alt text][image5]
![alt text][image6]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)

Here's a [link to my video result](./project_video.mp4)

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detection windows in each frame of the video.  From the positive detections, which have multiple overlaps, I created a heatmap by adding the window overlap together and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap and assumed each blob corresponded to a vehicle. With the blobs, I constructed single bounding boxes from the top left most coordinate and the bottom right most coordinate of the blob.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are two test images and their corresponding heatmaps:

![alt text][image7]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap on the frames displayed in the previous section
![alt text][image8]
![alt text][image9]
![alt text][image10]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image11]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Despite the SVM training on over 17000 samples and a 98.4% accuracy there were a significant amount of false positives. The railing on the bridge and parts of the center divider in the video had a lot of car-like features apparently. To address this, I averaged the bounding boxes for 20 frames (almost a second) and found that it reduced a large portion of the false positives along with smoothing out the bounding box changes from frame to frame for each actual car. The issue still remains that the false positives are there and why the classifier performed poorly on the video. I believe that a convolutional neural network would potentially perform better. I think that the HOG features can be similar to a car when looking at an intricate guard railing or tree shadow. There may be optimal HOG parameters and colorspaces that I did not explore thoroughly and with a convolutional neural network, fine-tuning these parameters may not be as big of an issue. I believe that the pipeline will immediately fail on any other road because there are over 3000 images that seem to have been extracted from the video included in the project. Therefore, with a 98.4% training accuracy (I say training accuracy because the test split of the data is still very similar to the training data) the SVM is potentially overfitted to the current stretch of road. This overfitting would be problematic if you move to another road with slightly different features because the training images are not representative of the new data. The frame smoothing makes the pipeline robust by eliminating the noise, but if there is large amounts of misclassification, the frame averaging will do nothing which is why I think that the classifier must be more accurate in the video. I believe it will be the same for the variation of window scaling. I found that despite adding various window scales, the misclassification of images with these increased number of window scales only slowed the pipeline and added more noise.
