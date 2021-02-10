# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 20:47:05 2020

@author: chait
"""

from blending_function import main_blend
from generate_mask import generate_mask
from rotate import rotate_vertically, rotate_left, rotate_right, flip_horizontally
from tiling import tile_pattern
from generate_cylinder import generate_cylinder 


def prepare_pattern_for_tiling(pattern_name, bigger_rectagle_width, bigger_rectangle_height, a =0, b = 0):
    
    #add_flipping_horizontal_code
        
    pattern_flipped_horizontally = 'pattern_flipped.jpg'
    flip_horizontally(pattern_name, pattern_flipped_horizontally)
    
    #？
    generate_mask(pattern_name,'masked512.jpg')
    main_blend(pattern_name,pattern_flipped_horizontally,'masked512.jpg','blended.jpg')
    
    
    rotate_vertically("blended.jpg")
    rotate_left("blended.jpg")
    rotate_left("flipped_vertical_blended.jpg")
    
    
    generate_mask('blended_rotated_vertical_rotated.jpg','mask_for_stage_2.jpg')
    main_blend('blended_rotated_vertical_rotated.jpg','flipped_vertical_blended_rotated_vertical_rotated.jpg','mask_for_stage_2.jpg','stage_1.jpg')
    
    
    rotate_right('stage_1.jpg','pattern_final.jpg')
    tile_pattern('pattern_final.jpg', bigger_rectagle_width, bigger_rectangle_height, 'final_pattern_tiled_full.png')

    #generate_cylinder('final_pattern_tiled_full.png', a, b)


#prepare_pattern_for_tiling(pattern_name='pattern_9.jpg', bigger_rectagle_width=5000, bigger_rectangle_height=5000, a=60, b=1)




