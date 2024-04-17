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

def img_matrix_fun():
    print("Hello from img_matrix.py!")


#Image Quality Matrix Comparison of Target and Ref images
class img_matrix:

  def psnr_fun(self, target,ref):
    #targer is HR 
    #ref is LR
    #assuming all images are rgbd

    self.target_data = target.astype(float)
    self.ref_data = ref.astype(float)

    #RMSE
    difference = self.ref_data - self.target_data

    difference = difference.flatten('C')

    rsme = math.sqrt(np.mean(difference ** 2. ))

    psnr = 20 * math.log10(255. / rsme)

    return psnr
    #low noise , high psnr score
    #high pnsr , better quality

    #define root mean square error

  def mse(self,target, ref):
    error = np.sum((self.target_data - self.ref_data)**2)
    error /= float(target.shape[0]*target.shape[1])
    return error 
    #higher mse , lower resoultion 

  #Combine all three image qualities
  def compare_images(self, target, ref):
    scores = []
    scores.append(self.psnr_fun(target,ref))
    scores.append(self.mse(target,ref))
    scores.append(ssim(target,ref,multichannel = True))#multichannel -> rgb
    #1 means completly same and 0 means different in structural similarity

    return scores
    







