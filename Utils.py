#training and testing of the facial keypoints detection model
import numpy as np
from ipynb.fs.full.GetAndPreprocessData import get_processed_train_and_test_data
from FacialKeypointsDetectionModel import train_face_detection_model,test_face_detection_model
from keras.models import load_model

def facial_keypoints_detection_model():
    train_image_data,train_keypoints,image_data_test=get_processed_train_and_test_data()
    #for the first time
    #model=train_face_detection_model(train_image_data,train_keypoints)
    model=load_model('face_detection_model.h5')
    test_face_detection_model(model,image_data_test)
    return model