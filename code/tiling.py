# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 21:36:58 2020

@author: chait
"""

from PIL import Image

def tile_pattern(src, bigger_rectagle_width, bigger_rectangle_height, final_pattern_save_name,grid_num=20):
    bg = Image.open(src)
    
    # The width and height of the background tile
    bg_w, bg_h = bg.size
    
    # Creates a new empty image, RGB mode, and size 1000 by 1000
    
    # The width and height of the new image
    w,h=bigger_rectagle_width,bigger_rectangle_height
    rate=grid_num*(bg_w*bg_h)/(w*h)
    rate=rate**(0.5)

    new_bgw,new_bgh=int(bg_w//rate),int(bg_h//rate)

    bg=bg.resize((new_bgw, new_bgh))
    w,h=(w//new_bgw)*new_bgw,(h//new_bgh)*new_bgh
    new_im = Image.new('RGB', (w, h))
    # Iterate through a grid, to place the background tile
    for j in range(0, h, new_bgh):
        for i in range(0, w, new_bgw):
            new_im.paste(bg, (i, j))
            
    new_im=new_im.resize((bigger_rectagle_width,bigger_rectangle_height))

    new_im.save(final_pattern_save_name,"PNG")