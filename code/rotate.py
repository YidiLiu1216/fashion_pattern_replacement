# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 21:10:48 2020

@author: chait
"""

import cv2

def rotate_vertically(originalImageSrc):
    
    originalImage = cv2.imread(originalImageSrc)
    flipVertical = cv2.flip(originalImage, 0)
    cv2.imwrite("flipped_vertical_blended.jpg",flipVertical)
    
    
    
def rotate_left(src):
    
    img1=cv2.imread(src)

    # rotate ccw
    out=cv2.transpose(img1)
    out=cv2.flip(out,flipCode=0)
    cv2.imwrite(src[:-4]+"_rotated_vertical_rotated.jpg", out)
    
    
    
def rotate_right(src, save_name):
    
    img = cv2.imread(src)
    img_rotate_90_clockwise = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite(save_name, img_rotate_90_clockwise)
    

def flip_horizontally(src, save_name):
    
    img = cv2.imread(src)
    flipHorizontal = cv2.flip(img, 1)
    cv2.imwrite("pattern_flipped.jpg",flipHorizontal)
    
    
    
    