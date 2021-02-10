# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 20:51:02 2020

@author: chait
"""


import cv2  # Not actually necessary if you just want to create an image.
import numpy as np


def generate_mask(orig_pattern_picture, mask_save_name):
    
    img = cv2.imread(orig_pattern_picture,0)
    height, width = img.shape[:2]
    
    blank_image = np.zeros((height,width,3), np.uint8)

    blank_image[:,0:int((0.9)*width)] = (0,0,0)      # (B, G, R)
    blank_image[:,int((0.9)*width):width] = (255,255,255)
    
    cv2.imwrite(mask_save_name,blank_image)

#generate_mask('C:/Users/chait/Desktop/image_warping/blending/miages/pattern_4/vertical/rotated_1.jpg')