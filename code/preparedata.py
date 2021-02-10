import os
import numpy as np
import cv2
from scipy import stats
from main import prepare_pattern_for_tiling
from image_distortion import distoration_image
from PIL import Image
import sys

color_in_seg = {226, 119, 169, 143, 221, 99, 95, 128, 186, 176, 199, 151}
new_seg = {95:95,10:169,60:99,126:128,210:119}
pattern_path=("pattern_png/pattern_8.png")

def preparemask(input_path,output_path):
    for original_image in os.listdir(input_path):
        image = cv2.imread(input_path + original_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for c in new_seg:
            mask_img = original_image[:-4] + "_{}.png".format(new_seg[c])
            mask=np.where(image==c,255,0)
            cv2.imwrite(output_path + mask_img, mask)

def preparepattern(labimg):
    hh, ww = labimg.shape[:2]
    newsize = (100, 100)
    bg = Image.open(pattern_path)
    bg = bg.resize(newsize)
    bg.save("pattern_9_resized.png")

    prepare_pattern_for_tiling(pattern_name='pattern_9_resized.png', bigger_rectagle_width=ww,
                               bigger_rectangle_height=hh)


    after_pattern = cv2.imread("final_pattern_tiled_full.png")
    after_pattern = cv2.cvtColor(after_pattern, cv2.COLOR_BGR2HSV)

    return after_pattern

def getcoordinates(mask):
    coord=np.where((mask == 255))
    coord = zip(coord[0],coord[1])
    coordinates=list(coord)

    mask=np.where((mask==255),1,0)
    r_mask=np.where((mask==1),0,1)
    return coordinates,mask,r_mask

def getmod(labimg, coordinates):
    sumbstrs = []
    for i, j in coordinates:
        sumbstrs.append(labimg[i, j])
    mod = stats.mode(sumbstrs)[0]
    maxb = max(sumbstrs)
    minb = min(sumbstrs)
    #return max(200,mod),maxb,minb
    return mod,maxb,minb

def prepareoutput(input_path,mask_path,output_path):
    for original_image in os.listdir(input_path):
        for c in color_in_seg:
          mask_img = original_image[:-4] + "_{}.png".format(c)
          if mask_img in os.listdir(mask_path):
              image = cv2.imread(input_path + original_image)
              image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
              mask = cv2.imread(mask_path+mask_img, cv2.IMREAD_GRAYSCALE)

              coord,mask,_=getcoordinates(mask)
              mod,maxb,minb=getmod(image[:,:,2],coord)
              image = np.int16(image)
              image=(image[:, :, 2]-mod)+200
              np.clip(image, 0, 255, image)
              image = np.uint8(image)

              mod, maxb, minb = getmod(image, coord)
              print(mod,maxb,minb)
              if (mod-minb)<200:
                  clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))
                  image = clahe.apply(image)


              img = image * mask

              cv2.imwrite(output_path + mask_img, img)

def prepareinput(input_path,mask_path,iuv_path,gray_scale_path,output_path):

    for original_image in os.listdir(input_path):
        for c in color_in_seg:
            mask_img = original_image[:-4] + "_{}.png".format(c)
            iuv = original_image[:-4] + "_IUV.png"
            if mask_img in os.listdir(mask_path) and mask_img in os.listdir(gray_scale_path):
                if iuv not in os.listdir(iuv_path):
                      iuv=""
                #print(input_path + original_image,mask_path+mask_img,gray_scale_path+mask_img,output_path + mask_img,iuv)
                image = cv2.imread(input_path + original_image)
                labimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                mask = cv2.imread(mask_path+mask_img, cv2.IMREAD_GRAYSCALE)

                result = cv2.imread(gray_scale_path+mask_img, cv2.IMREAD_GRAYSCALE)
                coordinates, mask, r_mask = getcoordinates(mask)
                mod, maxb, minb = getmod(result, coordinates)
                #bgrimage = fill_image_with_pattern_or_color(labimg, mask, result, c, iuv_path+iuv, mode='pattern')

                after_pattern = preparepattern(labimg)
                if iuv != "":
                    after_pattern = distoration_image(mask, c, after_pattern, iuv_path+iuv)
                after_pattern=np.int16(after_pattern)
                np.clip(after_pattern[:,:,2]+(np.int16(result)-mod),0,255,after_pattern[:,:,2])
                after_pattern = np.uint8(after_pattern)
                for i in range(3):
                    labimg[:, :, i] = after_pattern[:, :, i] * mask + labimg[:, :, i] * r_mask

                bgrimage = cv2.cvtColor(labimg, cv2.COLOR_HSV2BGR)

                cv2.imwrite(output_path + mask_img, bgrimage)
#preparemask("new_data/seg/","new_data/mask/")
#prepareoutput("sample_original_image/","new_data/mask/","new_data/out/"
'''
for i in range(31,32):
    pattern_path = ("pattern_png/pattern_{}.png".format(i))
    prepareinput("sample_original_image/","sample_mask_image/","IUV/","sample_output/","train_data/pattern_{}/".format(i))
'''
if __name__ == '__main__':
    s = sys.argv[1:]
    if len(s)>0:
      preparemask(s[0],s[1])
    else:
      preparemask("./data/seg/","./data/mask/")
