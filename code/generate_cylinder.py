# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 21:23:44 2020

@author: chait
"""

from wand.image import Image

def generate_cylinder(pattern_image, a, b):
    
    save_file_path= pattern_image[:-4]+"_cylinder.png"
    with Image(filename = pattern_image) as img:
        
        img.virtual_pixel = 'transparent'
        img.distort('plane_2_cylinder', (a,b)) 
        img.save(filename=save_file_path)


"""
generate_cylinder(pattern_image="blended_patterns_all/final_pattern_tiled_full_1.png", 
                  a=60, b=1)
"""
