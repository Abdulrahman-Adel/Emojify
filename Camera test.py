# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 09:36:57 2020

@author: Abdelrahman
"""

import cv2
from keras.models import model_from_json
from keras.preprocessing import image
import numpy as np


model = model_from_json(open("Emojify_model.json","r").read())

model.load_weights("Emojify_model.h5")


face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working!!!")
else:
    while True:
        
        ret, frame = cap.read()
        
        if ret:
            gray_frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
            face_detected = face_haar_cascade.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)
        
            for (x,y,w,h) in face_detected:
            
                cv2.rectangle(frame,(x,y-50),(x+w,y+h+10),(255,0,0),2)
                crop_gray = gray_frame[y:y+w,x:x+h]
                
                crop_gray=cv2.resize(crop_gray,(48,48))  
                img = image.img_to_array(crop_gray)  
                img = np.expand_dims(img, axis = 0)  
                img /= 255  
                
                predictions = model.predict(img)  
                max_index = int(np.argmax(predictions[0])) 
            
                emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')  
                predicted_emotion = emotions[max_index]  
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame,predicted_emotion,
                            (int(x),int(y)),font,1,(0,0,255),2,cv2.LINE_AA)
                
                resized_img = cv2.resize(frame, (1200, 860),
                                         interpolation = cv2.INTER_CUBIC) 
                cv2.imshow('Facial emotion analysis ',resized_img) 
                
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break
        else:
            print("Frame isn't working")
            print(cap.isOpened())
        
    cap.release()
    cv2.destroyAllWindows() 
            
            

        
    