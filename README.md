# **Behavioral Cloning**
---
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./readme/model.png "Model Visualization"
[image2]: ./readme/data_0.jpg "Training"
[image3]: ./readme/recover_0.jpg "Recovery Image 0"
[image4]: ./readme/recover_1.jpg "Recovery Image 1"
[image5]: ./readme/reverse.jpg "Reverse Image"

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* generator.py containing the script to create and train the model using a generator
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```
The file, drive.py has been altered to test the range of vehicle speed. The simulation has been tested at speeds 9, 20, and 30. All speeds tested has been successfully in not killing and virtual passengers. The video uses a speed of 30 mph to compress storage space and time. And like my good friend Ricky Bobby always says, "if you ain't first, you're last".

The scripts model.py and generator.py arrie at the same conclusion, however model.py will augment and store the input data. The script, generator.py, divides the input data into batches in order to save memory and has been added to the project submission to validate its understanding.

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Model Architecture.

The initial model included 3 convolutional layers, each proceeded by a RELU layer and max pooling layer. It then flattens, and goes through 4 fully connected layers. Although this model was sufficient in teaching the user how to proceed with the project, it was not able to accomplish a single loop.

The model used is a representation of the model developed by Nvidia. It includes cropping of the image to remove noise, normalization of the data, 5 convolutional layers proceeded by RELU layers, a flattening layer, and 4 fully connected layers.

#### 2. Attempts to reduce overfitting in the model

The model was initially trained with 10 epochs, but as the epochs increased, the validation loss increased. For the final model, the number of epochs was reduced to 3.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 85).

#### 4. Appropriate training data

In order to understand the important concepts of the project, I separated the training data into different folders and loaded it into the model iteratively. The data included 1 full lap, 1 reverse lap, recovery from imminent failure, extra data around the dirt roads, and the data provided by the project.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

When using the model provided by Nvidia, the validation loss was already quite low so I decided to persist with it with the data I had. Initially I used the two laps that I recorded myself. For some parts of the track, it handled things just fine, confirming that there was an impact with the model and data that the car was receiving.

The car was not driving in the middle, and it had problems  staying straight. I added recovery data so that the network would learn to adjust itself toward the middle, and the data files provided by the project. This would ensure that I had correct data in case I needed to change my model. The car ran better but had problems in certain areas where there was dirt texture instead of guidelines on the road.

I recorded my instances of the vehicle passing the dirt paths and the car was the successful in overcoming the problem areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 71-83) consisted of a convolution neural network with the following layers and layer sizes:

```
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to                     
====================================================================================================
cropping2d_1 (Cropping2D)        (None, 65, 318, 3)    0           cropping2d_input_1[0][0]         
____________________________________________________________________________________________________
lambda_1 (Lambda)                (None, 65, 318, 3)    0           cropping2d_1[0][0]               
____________________________________________________________________________________________________
convolution2d_1 (Convolution2D)  (None, 31, 157, 24)   1824        lambda_1[0][0]                   
____________________________________________________________________________________________________
convolution2d_2 (Convolution2D)  (None, 14, 77, 36)    21636       convolution2d_1[0][0]            
____________________________________________________________________________________________________
convolution2d_3 (Convolution2D)  (None, 5, 37, 48)     43248       convolution2d_2[0][0]            
____________________________________________________________________________________________________
convolution2d_4 (Convolution2D)  (None, 3, 35, 64)     27712       convolution2d_3[0][0]            
____________________________________________________________________________________________________
convolution2d_5 (Convolution2D)  (None, 1, 33, 64)     36928       convolution2d_4[0][0]            
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 2112)          0           convolution2d_5[0][0]            
____________________________________________________________________________________________________
dense_1 (Dense)                  (None, 100)           211300      flatten_1[0][0]                  
____________________________________________________________________________________________________
dense_2 (Dense)                  (None, 50)            5050        dense_1[0][0]                    
____________________________________________________________________________________________________
dense_3 (Dense)                  (None, 10)            510         dense_2[0][0]                    
____________________________________________________________________________________________________
dense_4 (Dense)                  (None, 1)             11          dense_3[0][0]                    
====================================================================================================
____________________________________________________________________________________________________
```

Here is a visualization of the architecture:

![alt text][image1]
###### Architecture Visualization

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving, and a second lap in the opposite direction.

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay in the middle of the lane. These images show what a recovery looks like starting from being to close the edge and steering back to the center.

![alt text][image3]
###### From the left side
<br><br>
![alt text][image4]
###### Back to the middle

This was done multiple times on various parts of the map.

To augment the data sat, I also flipped images and angles. This increased the input data by a factor of 2 without having to retrain.

I also ran the track backwards and flipped the images and angles. This gave the network even more data to work with.

![alt text][image5]
###### Reverse

After the collection process, I had roughly 10,0000 data points. I then preprocessed this data by cropping the top of the image containing the most noise, and adding a Lambda layer that handled normalization.

For efficiency, I removed 50% of the data where the car is driving straight. The controls were sensitive in the data recording, so in order to not record a non steering value when turning,

I finally randomly shuffled the data set and put 25% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3 as evidenced by the increased validation loss after 3 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
