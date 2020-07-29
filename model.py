# -*- coding: utf-8 -*-
"""
Created on Mon Jul 27 06:17:12 2020

@author: Abdelrahman
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

df = pd.read_csv("fer2013.csv")

def preprocessing_img(X):
    img_arr = []
    for i in X:
        i = list(i.split(" "))
        img_arr.append(np.reshape(i,(48,48,1)))
    img_arr = np.array(img_arr).astype(np.float32)
    img_arr = img_arr / 255.0
    return img_arr

img_arr_train = df["pixels"][df["Usage"]=="Training"]
img_arr_val = df["pixels"][df["Usage"]=="PublicTest"]

X_train = preprocessing_img(img_arr_train)
X_val = preprocessing_img(img_arr_val)

y_train = df["emotion"][df["Usage"]=="Training"]
y_val = df["emotion"][df["Usage"]=="PublicTest"]

y_train = pd.get_dummies(y_train)
y_val = pd.get_dummies(y_val)

model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(48, 48, 1)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.2))
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.3))
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam' ,
              metrics=['accuracy'])

gen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
it_train = gen.flow(X_train, y_train, batch_size=64)
steps = int(X_train.shape[0] / 64)

history = model.fit_generator(it_train, steps_per_epoch=steps, epochs=50, validation_data=(X_val,y_val), verbose=0)

model.save_weights("Emojify_model.h5")

Emojify_model = model.to_json()

with open("Emojify_model.json", "w") as json_file:  
    json_file.write(Emojify_model)  
    
    
    
plt.title('Cross Entropy Loss')
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='orange', label='test')

plt.title('Classification Accuracy')
plt.plot(history.history['accuracy'], color='blue', label='train')
plt.plot(history.history['val_accuracy'], color='orange', label='test')    

img_arr_test = df["pixels"][df["Usage"]=="PrivateTest"]
X_test = preprocessing_img(img_arr_test)
y_test = df["emotion"][df["Usage"]=="PrivateTest"]
y_test = pd.get_dummies(y_test)   

_, acc = model.evaluate(X_test,y_test)
 
  


        