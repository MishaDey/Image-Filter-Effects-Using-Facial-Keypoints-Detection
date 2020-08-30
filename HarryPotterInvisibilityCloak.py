import os
import cv2
import time
import numpy as np

def harry_potter_invisibility_cloak():
    print("\n\nEnter 'q' to exit.\n\n")
    capture=cv2.VideoCapture(0)
    time.sleep(3)
    iterr=0
    background_frame=0
    for index in range(60):
        ret,background_frame= capture.read()     
    #to reverse the order of content of background_pixel along axis 1
    background_frame=np.flip(background_frame,axis=1)
    
    while(capture.isOpened()):
        ret,frame=capture.read()
        if not ret :
            break
        iterr+=1
        frame=np.flip(frame,axis=1)
        #conversion of the RGB value to HSV 
        hsv_value=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        
        #generation of masks -- adjust ranges according to colour of cloak
        
        range_low=np.array([0,120,50])
        range_high=np.array([10,255,255])
        mask_1=cv2.inRange(hsv_value,range_low,range_high)
        
        range_low=np.array([170,120,70])
        range_high=np.array([180,255,255])
        mask_2=cv2.inRange(hsv_value,range_low,range_high)
        
        mask=mask_1+mask_2
        
        #morphologiction transformation -- Dilation of the image
        mask = cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask = cv2.morphologyEx(mask,cv2.MORPH_DILATE,np.ones((3,3),np.uint8))
        
        mask_new=cv2.bitwise_not(mask) #creation of inverted mask to segment out the 
        
        #segmenting out red colour part
        out1=cv2.bitwise_and(frame,frame,mask=mask_new)
        out2=cv2.bitwise_and(background_frame,background_frame,mask=mask)
        #show static background image for that segment
        #to get the weighted sum of arrays(out1 and out2)
        output=cv2.addWeighted(out1,1,out2,1,0)
        cv2.imshow("magic",output)
        
        key_pressed=cv2.waitKey(1) & 0xFF
        if key_pressed==ord('q'):
            break
    capture.release()
    cv2.destroyAllWindows()