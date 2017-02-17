# VehicleDetection
Vehicle detection project for SDC program

##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* ~~Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier~~
* ~~Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.~~ 
* ~~Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.~~
* ~~Implement a sliding-window technique and use your trained classifier to search for vehicles in images.~~
* ~~Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.~~
* ~~Estimate a bounding box for vehicles detected.~~

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4


####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

*see code cell 3 of ipython notebook

I began by reading in globs of positive and negative training data. I then used the provided utility batch feature extractor, which I modified slightly to use opencv image reading utilities, as well as to normalize all images to (0,1) to ensure no issues with format, etc. once they were in the pipeline. The feature selection consisted of (some combination of) HOG, spatial histogramming, and color binning.  

This provided an almost infinite space of parameter selection possibilities, which will be discussed below. 

Noteably, I used this feature extraction only for processing the training data. All other data for predictions was processed by a seperate function. This function, inspired by the tutorial video for the project, took

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

*see code cells 4, 5, and 6 of ipython notebook

For the classification stage of my pipeline, I worked with a number of candidate algorithms. Interestingly, though, the initial search was caused not by a need for superior classification accuracy, but by a strong desire for an algorithm capable of faster predictions. All of the classifiers I tested, including the first (SVM with a radial basis function kernel), produced 97%-99% classification accuracy. However, I found, using python's profile library and the ipython magix prun, that the speed of my pipeline was strongly hampered by the cost of the SVM-rbf's predict function (the linear kernel was somewhat better, but not enough). Over the course of taking tiles from a frame, the predict function was being called hundreds of times, for a total of often over 0.6 seconds per frame. Especially after the writing of the optimized HOG-compute function, this became the dominating cost in what was a 35 minute pipeline --simply miserable to experiment with for such a highly parameterized, empirical task. 

In search of a comparably accurate, but faster-to-classify algorithm, I tried SVM with a linear kernel, as well as XGBoost (as it is widely regarded as the top performing DT subtype at the present time). I found both of these to performant on the training set, but the linear kernel SVM was still quite slow (though it performed admirably on the video stream), and  XGBoost performed poorly on the actual training set relative to SVM despite it being tremendously fast, even without compilation of GPU capabilities or threading. Unfortunately the shear number of tuneable parameters made it unappealing to use under constrained time (though, were I to continue with this project not having found my final solution, I would certainly taken this as second best). Fortunately, I came across LinearSVC, I tremendously faster implementation of SVM that trained rapidly and performed as well as any of my other choices. 

Regarding choice of data for the classifier: I found that I was able to do without the customized bounding-box data Udacity provided more recently, though it would certainly find use if I had to produce a more robust algorithm. However, I found the extra negatives drawn from the project track added considerably to my pipeline's ability to keep false positives at a tolerable rate, and I thank you all for going through the trouble to add it! 

I also found it extremely useful to create a number of small clips for some (snooping-ish) testing as running on the whole video was simply too time-consuming for the sake of testing these algorithms --given that their performance on the training set was such a poor predictor, relatively, for performance on the video (with or without heightened regularization).

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

*see code cells 7 of ipython notebook

My sliding window search was implemented as a function in my CarTracker class. The various choices I made regarding it were based on three concerns: optimizing running time, supressing false positives, and allowing maximal flexibility. 

The first consideration I made was to slice off as much of the frame as possibile without losing accuracy. Obviously the sky and grass needn't be considered, and in our case, neither did the opposing direction of highway. This supressed the number of false positives and led to a more cleanly heat map. Moreover, in cutting out large swarths of the image, I was able to tile through the remaining image much more quickly, or alternatively, to allow for more tiling overlap at similar speeds. Obviously, trimming an image like this is a fairly empirical process, and decreases the robustness of the overall system, but similar savings could be found more reliably using a lane-finding algorithm to produce less "hand-tuned" estimates of where we need to be concerned with cars driving. 

On the topic of overlap, I found that considerable overlap in my tiling was extremely important (and that is actually what focused so much of my time on optimization, as high overlap increased run time extremely quickly). Allowing overlap ensured that each window was given a number of opportunities to classify the car as being present, and allowing the car to be spotted multiple times enabled the entire concept of the heatmap for false positive supression. Given the relatie weakness of linear models for image classification, false positives were essentially guaranteed; the heatmap/overlapping scheme essentially allowed us to run "multiple experiments" to find a better signal noise ratio. I found that much less thant 70% overlap was insufficient to find an acceptable S/N ratio in the images. 

In terms of window sizing, I found that using two, integer multiples of the training data shape for window sizes gave my pipeline some invariance to size, which was important considering the training data, which tended to be shot from a similar distance. I found that further limiting the portion of the image given to each window based on expected car size helped me (again) both to minimize computation time as well as false positive rate (running a window in a section of the frame where it was size mismatched with cars appearing in a similar location --again, this relates to what training data is used- only increased false positives on landscape, etc.). I think that a more robust solution would likely use a properly architected convnet capable of size invariance.

Finally, as mentioned previously, my windowing had to be entirely rewritten from the Udacity provided base solution. The hundreds of histograms run for each HOG extraction were the second of the pain points in running time. To bypass this, we were taught an alternative, in which we extracted HOG features from the entire frame and then tiled across that, rather than the opposite. This required a good bit of semi-frustrating arithmatic, as the HOG output was almost half a dozen dimensions, none of which were readily guessable to me ;-). The process included producing step counts and sizes, as well as some conversions of the original image based on a particular HOG frame bounds, to allow for spatial and color binning feature extraction. Extremely useful (though entirely unintuitive to me, at the outset) was the idea of scaling the input image and maintaining a constant window size. This required a bit of up and down scaling, both of the tiled images and their bounds, but ultimately made for a fairly logical means of dealing with various window sizes. All in all, this probably consumed the bulk of my time with this project, but the gains in speed were absolutely worthwhile, from easily 30-35 minutes (on a fairly beastly, brand new GPU workstation) down to a few minutes (for the full project video). 
![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
