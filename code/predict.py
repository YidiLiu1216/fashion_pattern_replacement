import model as m
import os
import cv2
import numpy as np
from main import prepare_pattern_for_tiling
from scipy import stats
import copy
from PIL import Image
from image_distortion import distoration_image
import sys

#parameter and path
image_size=512
type="pattern"
#the list includes cloth color in gray_scale segmentation images
color_in_seg = {226, 119, 169, 143, 221, 99, 95, 128, 186, 176, 199}
target_cloth_color = [[[65,105, 225]]]
#target_cloth_color = [[[255,255, 200]]]
model_path=("./model/")
model_name=("generator_clc_71_n.h5")
pattern_path=("./pattern_png/pattern_3.png")

input_path=("./data/input/")
mask_path=("./data/mask/")
output_path=("./data/output_gray/")
patterned_path=("./data/output_final/")
iuv_path=("./data/IUV/")

def gethsb(targetclr):
  targetclr=np.uint8(targetclr)
  targetclr=cv2.cvtColor(targetclr, cv2.COLOR_RGB2HSV)
  targetclr_list = np.ndarray.tolist(targetclr)
  _h,_s,_b=targetclr_list[0][0]
  return _h,_s,_b

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
    return max(200,mod),maxb,minb

def preparepattern(labimg):
    hh, ww = labimg.shape[:2]
    newsize = (100, 100)
    bg = Image.open(pattern_path)
    bg = bg.resize(newsize)
    bg.save("pattern_9_resized.png")

    prepare_pattern_for_tiling(pattern_name='pattern_9_resized.png', bigger_rectagle_width=ww,
                               bigger_rectangle_height=hh)


    after_pattern = cv2.imread("final_pattern_tiled_full.png")
    after_pattern = np.int16(cv2.cvtColor(after_pattern, cv2.COLOR_BGR2HSV))

    return after_pattern

def fill_image_with_pattern_or_color(labimg,mask,result,part=169,iuv="",mode=type,color=None):
    coordinates,mask,r_mask= getcoordinates(mask)
    cp_labimg = copy.deepcopy(labimg)

    mod,maxb,minb = getmod(np.int16(result), coordinates)
    ori_mod,ori_maxb,ori_minb = getmod(np.int16(labimg[:,:,2]), coordinates)
    rate=(ori_maxb-ori_minb)/max(1,(maxb-minb))
    rate=max(1,rate)

    cp_labimg = np.int16(cp_labimg)

    if mode == 'pattern':
      after_pattern=preparepattern(labimg)
      if iuv!="":
          after_pattern=distoration_image(mask,part,after_pattern,iuv)

      cp_labimg[:,:,0]=np.int16(after_pattern[:,:,0])
      cp_labimg[:,:,1]=np.int16(after_pattern[:,:,1])

      np.clip(np.int16(after_pattern[:,:,2]+((result-mod)*rate)), 0, 255,cp_labimg[:,:,2])
      #np.clip(np.int16(after_pattern[:, :, 2]) + (result - mod) , 0, 255, cp_labimg[:, :, 2])

    elif mode=='color':
        _h, _s, _b = gethsb(color)

        cp_labimg[:, :, 0] = _h
        cp_labimg[:, :, 1] = _s
        np.clip((result + _b-mod), 0, 255, cp_labimg[:, :, 2])
    cp_labimg=np.uint8(cp_labimg)

    for i in range(3):
        labimg[:, :, i] = cp_labimg[:, :, i] * mask+ labimg[:, :, i] * r_mask
    bgrimage = cv2.cvtColor(labimg, cv2.COLOR_HSV2BGR)

    return bgrimage

def load_model_keras():
  model = m.ResUNet(image_size)
  model.load_weights(model_path)
  return model

def predict_image(model,path,original_image,mask_path,c=169,iuv=""):
    image = cv2.imread(path + original_image)
    labimg = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ori_x, ori_y, _ = image.shape

    image = cv2.resize(image, (image_size, image_size))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask1 = cv2.resize(mask, (image_size, image_size))
    image = np.insert(image, 3, mask1, 2)
    image = image / 255.0
    image = np.array([image])

    result = model.predict(image)[0]
    result = result * 255.0
    result = cv2.resize(result, (ori_y, ori_x))

    bgrimage = fill_image_with_pattern_or_color(labimg, mask, result,c,iuv, mode='pattern',color=target_cloth_color)
    return result,bgrimage

def predict_original_image(model):
    for original_image in os.listdir(input_path):
        iuv_img=original_image[:-4]+"_IUV.png"
        for c in color_in_seg:
            mask_img = original_image[:-4] + "_{}.png".format(c)

            if mask_img in os.listdir(mask_path):
                if iuv_img in os.listdir(iuv_path):
                    iuv=iuv_path+iuv_img
                else:
                    iuv=""
                result,bgrimage=predict_image(model,input_path,original_image,mask_path+mask_img,c,iuv)
                cv2.imwrite(patterned_path + mask_img, bgrimage)
                cv2.imwrite(output_path + mask_img, result)


if __name__ == '__main__':
  s=sys.argv[1:]
  if len(s)>0 and len(s)<4:
     model_name=s[0]
     pattern_path=s[1]
  if len(s)==4:
     type=s[2]
     target_cloth_color=s[3]
  model_path=model_path+model_name
  model=load_model_keras()
  predict_original_image(model)

