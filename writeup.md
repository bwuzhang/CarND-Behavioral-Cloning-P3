**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/2.jpg "Original"
[image2]: ./examples/o_2.jpg "Lanes"
[image3]: ./examples/b_2.jpg "Brightness"
[image4]: ./examples/f_2.jpg "Flipped"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (net.py lines 67-71)

The model includes RELU layers to introduce nonlinearity for every convolution layer, and the data is normalized in the model using a Keras lambda layer (code line 64).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting.

The model was trained on 80% and validated on 20% of one data sets to ensure that the model was not overfitting (code line 82). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was set to 0.0001.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I applied 0.25 offset for right and left images, which equals to about 14 degrees.

For details about how I created the training data, see the next section.

I only used the data set provided by Udacity since I found they were large enough for me to get a reasonable result.
1. The first step to preprocess the data was to find lane in it using the pipeline I wrote in P1. Black lanes were drawn on all images.
![alt text][image2]
2. The next step was to horizontally flip all images and their corresponding steering measurements. This helped the algorithm learn turning right instead of turning left all the time.
![alt text][image4]
3. The final steps was applying random brightness change in the V channel of the images converted to HSV format.
![alt text][image3]

After all the steps, I tripled the number of training images I had.

The images were then cropped by 75 and 20 pixels on top and bottom respectfully in order to discard unnecessary information and reduce model size.
