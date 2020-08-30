import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import keras
from ipynb.fs.full.GetAndPreprocessData import plot_img_keypoints
from keras.models import Sequential
from keras.layers import (Conv2D,MaxPooling2D,Flatten,Dense,Dropout,Input)
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD,Adam
from skimage.io import imread
from skimage.transform import resize

def ConvolutionalNeuralNetworkModel():
    #CNN model -- Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64,kernel_size=3,strides=2,padding='same',input_shape=(96,96,1),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
   #output filters,kernel size - height and width of convolution window,strides-distance between consecutive applications of convolutional
    #filters,padding - "same"(o/p and i/p volume size matched) 
    model.add(Conv2D(128,kernel_size=3,strides=2,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Conv2D(128,kernel_size=3,strides=2,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Conv2D(64,kernel_size=1,strides=2,padding='same',activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2),padding='same'))
    model.add(Flatten())
    
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(30))
    return model

def Compile_Train_Save(model,train_image_data,train_keypoints):
    model.compile(loss='mean_absolute_error',optimizer='adam',metrics=['accuracy'])
    train_image_data=np.asarray(train_image_data).astype(np.float32)
    train_keypoints=np.asarray(train_keypoints).astype(np.float32)
    
    model.fit(train_image_data,train_keypoints,epochs=300,batch_size=100,verbose=1,validation_split=0.2)
    model.save('face_detection_model.h5')
    return model

def train_face_detection_model(train_image_data,train_keypoints):
    model=ConvolutionalNeuralNetworkModel()
    model=Compile_Train_Save(model,train_image_data,train_keypoints)
    return model
    
def test_face_detection_model(model,image_data_test):
    for i in range(0,20):
        input_image=np.reshape(image_data_test[i],(1,96,96,1))
        predicted_keypoints=model.predict(input_image)
        plot_img_keypoints(image_data_test[i],predicted_keypoints[0])
        