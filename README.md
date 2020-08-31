# Image Filter Effects Using Facial Keypoints Detection
#### 1. FACIAL KEYPOINTS DETECTION USING CONVOLUTIONAL NEURAL NETWORKS MODEL

##### Libraries and Modules used:

1. Keras

2. Numpy

3. Pandas

4. Matplotlib

5. Skimage




#### Dataset Description:

Facial Keypoints Detection Dataset by Dr Yoshua Bengio

https://www.kaggle.com/c/facial-keypoints-detection/ 

##### 1.1. Fetching and Data Preprocessing(Data Refining, Normalization, Augmentation) 

##### 1.2. Convolutional Neural Network Model Design and Training:

         Model Summary: https://drive.google.com/file/d/1U7O4FmUT6fxNmR8KpFfQ9BcZjpxJDZ3i/view?usp=sharing

##### 1.3. Testing the CNN Model for Facial Keypoints Detection
    Test Results : https://drive.google.com/file/d/14EvqX_bBJMHqjdnfg7E5JNGVOjwg2oZz/view?usp=sharing  
                
                Loss = 0.0.11 and Accuracy = 84%
              



#### 2. DESIGN FILTER EFFECTS USING THE CNN MODEL AND PROPER POSITIONING AND SCALING

       2.1. Popping Eyes Filter

       2.2. Moustache and glasses

       2.3. Devil Horns Filter

       2.4. Harry Potter Invisibility Cloak

       2.5. Easter Bunny Face Filter

       2.6. Dog Face Filter

       2.7. Santa Claus Filter

       2.8. Scary Jaw filter




##### Libraries and Modules used:

1. OpenCV

2. Numpy

3. Matplotlib

5. Skimage




##### PROCEDURE :

1. Capture the frame using OpenCV.

2. Use HaarCascade Classifier for frontal face detection in the captured frame.

3. Data preprocessing: Normalization and Refining.

4. The preprocessed data/image is sent to our CNN model for facial Keypoints Detection.

5. Use the output of the Facial Keypoints Detection model for scaling and positioning of our Filters




For Harry Potter Invisibility Cloak we just used OpenCV.

Capturing and Storage of background image or frame.Detection of the cloak(invisibility cloak with a particular colour) by setting the HSV(Hue Saturation Value) and use a segmentation algorithm to swap cloak coloured pixels with background pixels. That is, we swap the foreground frame with the background frame(generation of a mask).

HSV(Hue Saturation Value):

H (Hue) : (0-360 Degrees)
Red -> (0-60 degrees)
Magenta -> (301- 360 degrees)
Yellow -> (61-120 degrees)
Green -> (121-180 degrees)
Cyan -> (181-240 degrees)
Blue -> (241-300 degrees)

S (Saturation) :
The percentage of grey in a colour is called saturation of a colour.

V (Value) :
The percentage of brightness or intensity of a colour.




##### Demonstration Video :


