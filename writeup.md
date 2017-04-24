#**Traffic Sign Recognition** 

##Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/originalimages.jpg "originalimages"
[image3]: ./examples/augmentedimages.jpg "augmentedimages"
[image4]: ./examples/30kmh.jpg "Traffic Sign 1"
[image5]: ./examples/sliproad.jpg "Traffic Sign 2"
[image6]: ./examples/stopsign.jpg "Traffic Sign 3"
[image7]: ./examples/bumpyroad.jpg "Traffic Sign 4"
[image8]: ./examples/yield.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/peiyong86/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distributed.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I normalized the image data so that data have mean 0 and variance of 1.

I decided to generate additional data because data augmentation can reduce over-fitting problem and boost the performance.

To add more data to the the data set, I used the following techniques from this paper: http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

The data augmentation techniques I used include: rescale, shift and rotate.
I didn't use flip because some trafic sign may have fixed direction.

Here is an example of an original image and an augmented image:
![alt text][image2]
![alt text][image3]

Instead generate a augmented data set, I apply the augmentation function on every batch of data.
I did this because of two reasons: first, to generate more different data. Second, to use less memory.
The disadvantage is that it will increase computation cost. 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x12 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x12   				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x24	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x24   				|
| Convolution 2x2	    | 1x1 stride, valid padding, outputs 4x4x48 	|
| RELU					|												|
| Flatten				|input conv2									|
| Flatten				|input conv3									|
| Concat				|concate above 2 flat layer						|
| Dropout				| 0.8											|
| Fully connected		| inputs 1368, outputs 512  					|
| RELU					|												|
| Dropout				|												|
| Fully connected		| inputs 512, outputs 256  						|
| RELU					|												|
| Dropout				| 0.8											|
| Fully connected		| inputs 256, outputs num of classes  			|
| Softmax				| etc.        									|
|						|												|
|						|												|
 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer, with following parameters:
learning rate: 0.0005
batch size: 128
epoch num: 100
drop out keep prob: 0.8

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
At first I choosed the LeNet-5 architecture, because it was introduced in udacity's course.
The images size of traffic sign data is same as input size of LeNet-5.

* What were some problems with the initial architecture?
The model is not deep engouth, the initial architecture is under-fitting on our data.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
After I tried LeNet-5, I adjust the model architecture according to the paper http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf.
The new architecture has move convolution layers, and concat output from different convolution layers as the input to fully conected layers.

* Which parameters were tuned? How were they adjusted and why?
I have increated the epoch number because the new model have more weights, thus need more iterations to train.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
The convolution layer is translation invariant, so it works well on images. Because our sample images can have different size, location and oritaion.
Dropout layer forces the model to learn a redundant representation and prevents over-fitting.

If a well known architecture was chosen:
* What architecture was chosen?
The architecture from http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
* Why did you believe it would be relevant to the traffic sign application?
Because the paper reports really good results on traffic sign application.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
If three accuracy is high, it evidences that the model does not over-fitting or under-fitting, and have good generalize ability.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The forth image might be difficult to classify because the sign image is rotated a lot and the image ratio is quite different from training data.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 30 km/h	      		| 30 km/h   					 				|
| Slippery Road			| Slippery Road      							|
| Stop Sign      		| Stop sign   									| 
| Bumpy road  			| Bicycles crossing 							|
| Yield					| Yield											|




The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares unfavorably to the accuracy on the test set of 96%.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 14th cell of the Ipython notebook.

For the first image, the model is absolutely sure that this is a 30 km/h sign (probability of 0.99...), and the image does contain a 30 km/h sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .9999...    			| 30 km/h   									| 
| 1e-14     			| 50 km/h 										|
| 3e-22					| Bicycles crossing								|
| 5e-23	      			| 60 km/h   					 				|
| 2e-23				    | end of 80 km/h     							|


For the second image, the third image and the fifth image, the model is also quite sure about the result( probability >= 0.9998)

For the forth image (the incorrect predict one), the probability is relatively lower (probability of 0.978) compare to other images.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .978       			| Bicycles crossing   							| 
| 0.02      			| Bumpy road 									|
| 2e-14					| Wild animals crossing							|
| 1e-14	      			| Wild animals crossing   		 				|
| 3e-15				    | No passing for vehicles over 3.5 metric tons	|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


