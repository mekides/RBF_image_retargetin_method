# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 12:49:55 2019
@author: mekides Assefa Abebe
@place:NTNU - ColourLab
"""
import cv2
import numpy as np
from RBF_Local_utils import *
from main_RBF import main_RBF
#file locations
img_path = 'inputs/images/'
mask_path = 'inputs/masks/'
output_path = 'outputs/'

#Read the input image and its corresponding region of interest mask
img = cv2.imread(img_path + '2.png')
msk = cv2.imread(mask_path + '2.png_msk.png',cv2.IMREAD_GRAYSCALE)
#Since my ROI image is grayscale image, we need to convert it to binary image
#first
ret,bmask = cv2.threshold(msk,2,255,0)

#define the parameters
r = np.array([0.6,0.8]) # the scaling factor output_size = input_size * r
epislon = 0.99 # The strength parameter, takes values in the interval (0,1)
Es = 30 #mesh element size in pixels (single element = Es x Es pixels) 
 
img_out,mesh = main_RBF(img,r,bmask,epislon,Es)

cv2.imwrite(output_path + '2_retargeted_image.png',img_out)
cv2.imwrite(output_path + '2_deformed_mesh.png',mesh)
cv2.imshow('Output',img_out)
cv2.imshow('Final mesh',mesh)

print("Press any key to stop excution.")
if cv2.waitKey(0):
   cv2.destroyAllWindows()