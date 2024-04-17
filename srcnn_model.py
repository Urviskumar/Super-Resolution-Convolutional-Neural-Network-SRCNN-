import math
import numpy as np
from typing import Dict, List, Optional
import pandas as pd
from matplotlib import pyplot as plt

import keras
from keras.models import Sequential 
from keras.layers import Conv2D, Input
from keras.optimizers import Adam , SGD
from keras.callbacks import ModelCheckpoint

import sys
import cv2
import matplotlib
import skimage
import os 
import time

from skimage.metrics import structural_similarity as ssim
from h5py import filters
import tensorflow


def model():

    #define model type - sequential
    srcnn = Sequential()

    #add model layers
    srcnn.add(Conv2D(filters=128, kernel_size = (9,9), kernel_initializer='glorot_uniform', 
                     activation='relu', padding='valid',use_bias=True,input_shape = (32,32,1)))
    
    srcnn.add(Conv2D(filters = 64, kernel_size= (3,3), kernel_initializer= 'glorot_uniform',
                      activation='relu', padding='same', use_bias=True))

    srcnn.add(Conv2D(filters=1,kernel_size=(5,5), kernel_initializer='glorot_uniform',
                      activation='linear',padding='valid',use_bias= True))


    #Optimizer
    adam = Adam(lr = 0.0003)

    #Compile the Model
    srcnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return srcnn

def prediction_model():

    #define model type - sequential
    srcnn = Sequential()

    #add model layers
    srcnn.add(Conv2D(filters=128, kernel_size = (9,9), kernel_initializer='glorot_uniform', 
                     activation='relu', padding='valid',use_bias=True,input_shape = (None,None,1)))
    
    srcnn.add(Conv2D(filters = 64, kernel_size= (3,3), kernel_initializer= 'glorot_uniform',
                      activation='relu', padding='same', use_bias=True))

    srcnn.add(Conv2D(filters=1,kernel_size=(5,5), kernel_initializer='glorot_uniform',
                      activation='linear',padding='valid',use_bias= True))


    #Optimizer
    adam = Adam(lr = 0.0003)

    #Compile the Model
    srcnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mean_squared_error'])

    return srcnn


from sklearn.model_selection import train_test_split

def train(data, labels):
    srcnn_model = model()

    # Split the data into training and validation sets
    data_train, data_val, label_train, label_val = train_test_split(data, labels, test_size=0.2, random_state=42)

    # Convert lists to numpy arrays, if they are not already
    data_train = np.array(data_train)
    label_train = np.array(label_train)
    data_val = np.array(data_val)
    label_val = np.array(label_val)
    print("Data train shape:", data_train.shape)
    print("Label train shape:", label_train.shape)
    print("Data val shape:", data_val.shape)
    print("Label val shape:", label_val.shape)

    checkpoint = ModelCheckpoint("SRCNN_check.h5", monitor='val_loss', verbose=1, save_best_only=True,
                                 save_weights_only=False, mode='min')
    callbacks_list = [checkpoint]

    history=srcnn_model.fit(data_train, label_train, batch_size=128, validation_data=(data_val, label_val),
                    callbacks=callbacks_list, shuffle=True, epochs=500, verbose=0)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Epochs vs. Loss')
    plt.legend()
    plt.show()
    srcnn_model.save('srcnn_model.h5')

    return srcnn_model

def modcrop(img, scale):
    tmpsz = img.shape #temp size
    size= tmpsz[0:2]
    size=size-np.mod(size,scale)#remainder 
    img = img[0:size[0], 1:size[1]]
    return img 

def shave(image, border):
    img = image[border: -border, border: -border]
    return img
#Prediction 

def predict5(image_path, model_weights_path):

    #load thee weights with model-srcnn
    srcnn = prediction_model()
    srcnn.load_weights(model_weights_path)

    #load degraded image and Ref images to compare after doing HR
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread('/content/drive/My Drive/CV/Project_SR/Set5/{}'.format(file))

    #preprocess
    ref = modcrop(ref,3)
    degraded = modcrop(degraded,3)

    #convert the image into YCrCb  - srcnn trained on Y channels not RGB
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb) #first B - G - R syntax
    
    #image slice and normalize 
    Y = np.zeros((1,temp.shape[0],temp.shape[1],1), dtype = float)
    Y[0, : , : , 0] = temp[: , : ,0].astype(float)/ 255 #normalize between 0 and 1
    #perfoem SRCNN

    pre = srcnn.predict(Y, batch_size= 1)
    
    #post 
    pre *=255
    pre[pre[:]> 255]= 255 #if pixel is more than 255 , consider it 255
    pre[pre[:]<0] = 0 # if pixel is less than 0 consider it 0
    pre = pre.astype(np.uint8) # pixel are still in float values, convert it into integer 

    # above is Y channel we want in rgb
    temp = shave(temp, 6) # crop border of 6 , 
    temp[:,:,0] = pre[0,:,:,0] #keeping Cr and Cb differences channels,Copying Y channel into temp 
    output = cv2.cvtColor(temp,cv2.COLOR_YCrCb2BGR)

    #REMOVE BORDER 
    ref = shave(ref.astype(np.uint8),6)
    degraded = shave(degraded.astype(np.uint8),6)

    # all references and degraded is in same size now

    # return images
    return ref, degraded, output

def predict14(image_path, model_weights_path):

    #load thee weights with model-srcnn
    srcnn = prediction_model()
    srcnn.load_weights(model_weights_path)

    #load degraded image and Ref images to compare after doing HR
    path, file = os.path.split(image_path)
    degraded = cv2.imread(image_path)
    ref = cv2.imread('/content/drive/My Drive/CV/Project_SR/Set14/{}'.format(file))

    #preprocess
    ref = modcrop(ref,3)
    degraded = modcrop(degraded,3)

    #convert the image into YCrCb  - srcnn trained on Y channels not RGB
    temp = cv2.cvtColor(degraded, cv2.COLOR_BGR2YCrCb) #first B - G - R syntax
    
    #image slice and normalize 
    Y = np.zeros((1,temp.shape[0],temp.shape[1],1), dtype = float)
    Y[0, : , : , 0] = temp[: , : ,0].astype(float)/ 255 #normalize between 0 and 1
    #perfoem SRCNN

    pre = srcnn.predict(Y, batch_size= 1)
    
    #post 
    pre *=255
    pre[pre[:]> 255]= 255 #if pixel is more than 255 , consider it 255
    pre[pre[:]<0] = 0 # if pixel is less than 0 consider it 0
    pre = pre.astype(np.uint8) # pixel are still in float values, convert it into integer 

    # above is Y channel we want in rgb
    temp = shave(temp, 6) # crop border of 6 , 
    temp[:,:,0] = pre[0,:,:,0] #keeping Cr and Cb differences channels,Copying Y channel into temp 
    output = cv2.cvtColor(temp,cv2.COLOR_YCrCb2BGR)

    #REMOVE BORDER 
    ref = shave(ref.astype(np.uint8),6)
    degraded = shave(degraded.astype(np.uint8),6)

    # all references and degraded is in same size now

    # return images
    return ref, degraded, output
























