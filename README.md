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
![alt text][training]

[training]: ./supporting_media/positivenegative.png
[hog_features]: ./supporting_media/HOGfeatures.png
[pipeline0]: ./supporting_media/boxes_on_image0.png
[pipeline1]: ./supporting_media/boxes_on_image1.png
[pipeline2]: ./supporting_media/boxes_on_image2.png
[pipeline3]: ./supporting_media/boxes_on_image3.png
[pipeline4]: ./supporting_media/boxes_on_image4.png
[pipeline5]: ./supporting_media/boxes_on_image5.png

[video1]: ./project_video.mp4


####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

*see code cell 3 of ipython notebook, as well as bottom where provide utilities are defined.

I began by reading in globs of positive and negative training data. I then used the provided utility batch feature extractor, which I modified slightly to use opencv image reading utilities, as well as to normalize all images to (0,1) to ensure no issues with format, etc. once they were in the pipeline. The feature selection consisted of (some combination of) HOG, spatial histogramming, and color binning.  

This provided an almost infinite space of parameter selection possibilities, which will be discussed below. 

Noteably, I used this feature extraction only for processing the training data. All other data for predictions was processed by a seperate function.

See sample of training data below:

![alt text][training]

####2. Explain how you settled on your final choice of HOG parameters.

Regarding parameters, I found ultimately that the defaults provided we're mostly adequate. 

* I experimented somewhat with colorspace, finding YCrCb, HSV, and HLS to work about comparably. I am, however, more comfortable with HSV as far as intuitions as I've been using it all semester; and, it is also well regarded for CV generally, but also in terms of describing light cast, which seems to have been a recurring issue for many students, (issues that I've largely been spared having used HSV carefully). Moreover, I tried playing with the number of channels, and while I still feel that the hue channel is superfluous, and possibly detrimental for general, robust use, the pipeline worked well enough (and with LinearSVC, fast enough) that I kept it just in case, and maybe for a bit of additional accuracy at the cost of reduced robustness. 
* Block size is widely accepted as more-or-less settled on 2x2 (and in fact, I believe some major implementations allow for nothing larger). 
* Orientation count is also more or less ideal at 9 (that 9 is typically the peak of improvement before actual regression can even be found in the literature).
* Pixels per cell I did attempt to double, but I found I had been slightly better off with 8.
* I found considerable improvement by including color binning features regarding the black vs white cars, which were not recognized even nearly equally well with just HOG features, and also some with spatial. Moreover, I found doubling the bin size for colors added some without considerably altering feature size. I also had normalized my images to (0,1), so my binning range changed as well.

See below for the intensity image and HOG features for each channel in an HSV mapping of the above positive example using my final results (8 px/cell, 9 orientations, 2 cells per block, 32 bin color, 32x32 spatial:

![alt text][hog_features]

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

*see code cells 4, 5, and 6 of ipython notebook

For the classification stage of my pipeline, I worked with a number of candidate algorithms. Interestingly, though, the initial search was caused not by a need for superior classification accuracy, but by a strong desire for an algorithm capable of faster predictions. All of the classifiers I tested, including the first (SVM with a radial basis function kernel), produced 97%-99% classification accuracy. However, I found, using python's profile library and the ipython magix prun, that the speed of my pipeline was strongly hampered by the cost of the SVM-rbf's predict function (the linear kernel was somewhat better, but not enough). Over the course of taking tiles from a frame, the predict function was being called hundreds of times, for a total of often over 0.6 seconds per frame. Especially after the writing of the optimized HOG-compute function, this became the dominating cost in what was a 35 minute pipeline --simply miserable to experiment with for such a highly parameterized, empirical task. 

In search of a comparably accurate, but faster-to-classify algorithm, I tried SVM with a linear kernel, as well as XGBoost (as it is widely regarded as the top performing DT subtype at the present time). I found both of these to performant on the training set, but the linear kernel SVM was still quite slow (though it performed admirably on the video stream), and  XGBoost performed poorly on the actual training set relative to SVM despite it being tremendously fast, even without compilation of GPU capabilities or threading. Unfortunately the shear number of tuneable parameters made it unappealing to use under constrained time (though, were I to continue with this project not having found my final solution, I would certainly taken this as second best). Fortunately, I came across LinearSVC, I tremendously faster implementation of SVM that trained rapidly and performed as well as any of my other choices. 

As far as training specific data processing, images were scaled to (0,1) and passed to `sklearn.preprocessing.StandardScaler()` for normalization. The data was also shuffled and then split into training and test sets on an 80:20 split. Knowing that the data was largely aquired consecutively would be a serious concern in an attempt to build a more robust classifier, but in this case, it seems to be acceptable to ignore that fact in terms of success on the given video. 

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

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Generally, I optimized on window size, colorspace, learning algorithm, and a few other parameters. These are all covered in detail in their respective sections above. 

Specific to the classifier, my traditional 'tuning' consisted on learning family selection, strength of regularization, and data preprocessing. This is all described in detail in the above section on the selection of and implementation of my classifier. Again though, the bulk of my trouble was not in getting the classifier to recognize a car, but in getting a (now) less-than linear classifier to turn up few enough false positives to allow my heatmap method to deal with them. Chiefly, this consisted of providing training-data-like perspectives to the classifier (described above in section on choice and positional range of different window sizes), incorporating both chromatic and contour data, and presenting each window with a number of slightly translated tiles of the cars via tile overlap in the feature extractor. Beyond this, I found my time better spend optimizing the heatmap/FP-rejection portion of my pipeline.  

A number of examples using the sample images I worked with for the non-video part of development, using one of my more successful paramter sets for stills (that was, incidentally, subpar for video), 1, 1.5 scale windows, rather loose boundaries:

![alt text][pipeline0]
![alt text][pipeline1]
![alt text][pipeline2]
![alt text][pipeline3]
![alt text][pipeline4]
![alt text][pipeline5]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./supporting_media/final_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

For each positive result, I incremented the corresponding space in my heatmap by one. After passing all windows through for a frame, I thresholded the frame at a medium single-digit integer, which I tuned purely empirically. I then pushed this new heatmap into a quasi-lowpass filter with the previous rolling average heatmap (it's parameter alpha was, again, tuned empirically), and finally, I took a last minute cut, throwing away any newly introduced, low intensities. I then passed this final composite map into `scipy.ndimage.measurements.label()` which conveniently yielded me labels and patches corresponding to raised intensities. I then assumed each blob corresponded to a vehicle (if continued, I had had some conversation with classmates about mixed results in using particle filters to main individuality of adjacent vehicles (next semester, maybe?!). I finally wrote bounding boxes to the image via OpenCV's rectangles.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][video_frams]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I mentioned this a few times before, but it merits repeating: the time required for the default provided implementations was just unworkable in terms of the kind of empiricism this project requires. By far the bulk of my time was spent finding and/or implementing faster ways to deal with HOG and SVM. The secondary problem was the general insufficiency of SVM for image classification compared to modern methods. Obivously it was the go-to for a number of years, but it's hard to see the kind of results a poorly tuned CNN provides and then go back to basically any linear method. The result of using a second best classifier was a considerable amount of full-on and partial (calling a very small part of the car "car" added considerable noise heat) false positives. 

Personally, I was more or less able to get past the slowness, the HOG and LinearSVC made processing tolerable (although not at all if you were to want to use this for real time). But the jitteriness of my boxes still belay my troubles with false positives, I discuss a few larger/longer-term ideas for fixing this, but the best best in the short term would likely be continued focus on feature selection and on optimizing the filter/thresholding/averaging aspects of my heatmap function. 

Among those longer term solutions: adding robustness here pretty much demands a CNN, and one architected to emphasize scale invariance. An improved dataset, with more conditions would help; and augmentation of the data is the obvious low-hanging fruit here. More advanced signal processing and/or clustering for the sake of specific vehicle tracking despite obfuscation, etc. would also help.

The pipeline as it stands (and this seems common) doesn't instill much confidence. The image is sliced specific to this camera positioning and angle, the classifier is focused on same-direction-of-traffic vehicle positioning, and the data seems almost entirely focused on sedans, on precipitation free, well-lit days. Moreover, all of the empiricism involved surely bled bits of test into the training that wouldn't even cross my mind. 

At the end of the day, I've really learned one thing: we have it much better now than our counterparts did even 5-10 years ago. The staff collected and augmented data and the students slaved for collective months over a hot workstation, tuning everything possible, and the best of our results is still second-best to even the low-grade NN attempts. Pure CV with old-school (in terms of popularity) ML hypothesis families is just hard. Real hard. 
