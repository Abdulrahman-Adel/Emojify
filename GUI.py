# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 15:58:45 2020

@author: Abdelrahman
"""
import tkinter as tk
from tkinter import *
from PIL import ImageTk, Image
import cv2
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

cv2.ocl.setUseOpenCL(False)

model = model_from_json(open("Emojify_model.json","r").read())
model.load_weights("Emojify_model.h5") 

emoji_dist={0:"./emojis/angry.jpg",1:"./emojis/disgusted.jpg",
                2:"./emojis/fearful.jpg",3:"./emojis/happy.jpg",
                4:"./emojis/neutral.png",5:"./emojis/sad.jpg",6:"./emojis/surprised.jpg"}

index_num = 0

def cam_vid():
    face_haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml") 
    cap = cv2.VideoCapture(0)
    
    ret, frame = cap.read()
       
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
        index_num = max_index
        
    
    if ret:        
        resized_img = cv2.resize(frame, (400, 400),
                                 interpolation = cv2.INTER_CUBIC)     
        cv2image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        Label1.imgtk = imgtk
        Label1.configure(image=imgtk)
        Label1.after(10, cam_vid)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
         exit()

 

def emojy_pic():
    emojy = cv2.imread(emoji_dist[index_num])
   
    resized_img = cv2.resize(emojy, (400, 400),
                                              interpolation = cv2.INTER_CUBIC)  
    pic2=cv2.cvtColor(resized_img,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(pic2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    Label2.imgtk2=imgtk2
    Label2.configure(image=imgtk2)
    Label2.after(10, emojy_pic)
   
if __name__ == "__main__":
    
    root = tk.Tk()

    root.title("Emojify")
    root.geometry("1100x600")
    root.configure(bg="black")

    def close_window():
        root.destroy()

    title = tk.Label(root,text="Facial Emotions Regocnition",
                     font="Times 32",anchor="center",fg="gray",width=25,height=1)
    title.pack()

    Label1 = tk.Label(root)
    Label1.pack(padx = 50,side = tk.LEFT)

    Label2 = tk.Label(root)
    Label2.pack(padx = 50,side = tk.LEFT)

    button = tk.Button(root,text="Exit",bg="gray",command=close_window,anchor="center",width=4,height=1)
    button.pack(side = tk.BOTTOM)
    
    cam_vid()
    emojy_pic()

 #   cap.release()
    root.mainloop()