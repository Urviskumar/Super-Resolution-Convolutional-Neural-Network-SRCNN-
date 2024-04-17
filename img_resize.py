import math
import numpy as np
from typing import Dict, List, Optional
import keras
from keras.models import Sequential 
from keras.layers import Conv2D, Input
from keras.optimizers import Adam , SGD

import sys
import cv2
import matplotlib
import skimage
import os 
import time
import matplotlib.pyplot as plt

from skimage.metrics import structural_similarity as ssim

def img_resize_fun():
    print("Hello from img_resize.py!")


#Creating low resolution matrix from img_resize
#Data set has set 5 + set 14 only rgb images
class img_resize:

    def prepare_images(self, path, factor):
        images = []
        filenames = []
        for file in os.listdir(path):
            img = cv2.imread(path + '/' + file)
            height, width, channels = img.shape
            new_height = int(height / factor)
            new_width = int(width / factor)
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
            img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
            images.append(img)
            filenames.append(file)
        return images, filenames

'''
  def prepare_images(self, path,factor):
    #path to the image file 
    #factor by which want to downgrade 2 or 3 or 4
    #assuming all images are rgbd
    for file in os.listdir(path):
      #open the file

      img = cv2.imread(path + '/' + file)

      #find old and new images dimensions

      height , width, channels = img.shape

      new_height = int(height / factor) # divide the img size by 2 if factor is 2
      new_width = int(width / factor)
      
      # resize the image - downsampling
      img = cv2.resize(img, (new_width, new_height), interpolation= cv2.INTER_LINEAR)

      #resize the image - upsampling 
      img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
#This resize will not change the same but it will reduce the resolution
      
    return img 
    
      # save the image
      save_dir = '/content/drive/My Drive/CV/Project_SR/LR_Images'#this will create 100 LR img for training as of now
      if not os.path.exists(save_dir):
      # If the directory does not exist, create it
         os.makedirs(save_dir)
      print('Saving  {}'.format(file))
      cv2.imwrite(os.path.join(save_dir, file), img)  # save LR into images directory'''




      
















