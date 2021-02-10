import thinplate as tps
import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
from PIL import Image
from main import prepare_pattern_for_tiling
from blending_function import blend_image

pattern_path=("pattern_png/pattern_3.png")

top={95,169,221,128,186,176,199}
under={99,119}
#top_body=[2,9,10,3,4]
top_body=[2,1,9,10,5,6]
under_body=[7,8]
v_flip={15,16,7,8,12,11,10,9,2}
u_flip={20,19,9,10,11,12,8}
#v_add={16,20,15,19,1,7,8,11,12}
v_add={16,20,15,19,7,8,11,12}
#u_add={20,22,19,21,13,14,11,12}
u_add={13,14,11,12}
#body_merge={1:2,11:7,9:7,13:7,12:8,10:8,14:8,18:9,16:9,20:9,22:9,15:10,17:10,19:10,21:10}
body_merge={11:7,9:7,13:7,12:8,10:8,14:8,18:9,16:9,20:5,22:5,15:10,17:10,19:6,21:6}
grid_num=5

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
def draw_new_coutour(im,IUV):
    plt.imshow(im)
    plt.contour(IUV[:, :, 1] / 256., 10, linewidths=1)
    plt.contour(IUV[:, :, 2] / 256., 10, linewidths=1)
    plt.axis('off');
    plt.show()
    return
def warp_image_cv(img, c_src, c_dst, dshape=None):
    dshape = dshape or img.shape
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grid = tps.tps_grid(theta, c_dst, dshape)
    mapx, mapy = tps.tps_grid_to_remap(grid, img.shape)
    return cv2.remap(img, mapx, mapy, cv2.INTER_CUBIC)
def merge_body(iuv):
    for i in range(1,25):
       mask=np.where(iuv[:,:,0]==i,1,0)
       r_mask=np.where(mask==1,0,1)
       part=copy.deepcopy(iuv)
       for a in range(3):
           part[:,:,a]=iuv[:,:,a]*mask
       part[:,:,1]=part[:,:,1]/2.0
       part[:,:,2]=part[:,:,2]/2.0
       if i in u_flip:
           part[:,:,1]=128.0-part[:,:,1]
       if i in v_flip:
           part[:,:,2]=128.0-part[:,:,2]
       if i in u_add:
           part[:,:,1]=128.0+part[:,:,1]
       if i in v_add:
           part[:,:,2]=128.0+part[:,:,2]
       if i in body_merge:
           part[:,:,0]=body_merge[i]

       for a in range(3):
           iuv[:,:,a]=part[:,:,a]*mask+iuv[:,:,a]*r_mask

    return iuv
def get_box(mask):
    coord = np.where((mask == 1))

    max_y=max(coord[0])
    min_y=min(coord[0])
    max_x=max(coord[1])
    min_x=min((coord[1]))
    return max_x,min_x,max_y,min_y
def out_body_part(body,part,x,y):
    body_mask=np.ones((x,y),dtype="uint8")
    part_mask=np.zeros((x,y),dtype="uint8")

    body_array=np.int32([[body[1],body[3]],[body[1],body[2]],[body[0],body[2]],[body[0],body[3]]])
    part_array=np.int32([[part[1],part[3]],[part[1],part[2]],[part[0],part[2]],[part[0],part[3]]])
    cv2.polylines(part_mask, np.int32([part_array]), 1, 1)
    cv2.fillPoly(part_mask, np.int32([part_array]), 1)

    cv2.polylines(body_mask, np.int32([body_array]), 1, 0)
    cv2.fillPoly(body_mask, np.int32([body_array]), 0)


    result=part_mask*body_mask

    return result
def getsource_grid(mask):

    box=get_box(mask)
    w,h=mask.shape
    max_x,min_x,max_y,min_y=box[0],box[1],box[2],box[3]
    #print(max_x,min_x,max_y,min_y)
    grid_x=np.linspace(min_x,max_x,grid_num)
    grid_y=np.linspace(min_y+10,max_y-10,grid_num)

    x,y=np.meshgrid(grid_x,grid_y)
    #x,y=x/h,y/w
    x,y=x+(h//2),y+(w//2)
    grid = zip(x.flatten(), y.flatten())
    grid = np.array(list(grid))
    return grid,box
def getdes_grid(mask,iuv,source,test=False):

    w,h=mask.shape
    part_iuv=copy.deepcopy(iuv)
    part_u=part_iuv[:,:,1]*mask
    part_v=part_iuv[:,:,2]*mask
    tmp_u=copy.deepcopy(part_u)
    tmp_v=copy.deepcopy(part_v)
    tmp_u.flatten()
    tmp_v.flatten()
    max_u=np.max(tmp_u)
    max_v=np.max(tmp_v)
    tmp_u=np.where(tmp_u==0,256,0)+tmp_u
    tmp_v=np.where(tmp_v==0,256,0)+tmp_v
    min_u=np.min(tmp_u)
    min_v=np.min(tmp_v)
    diff_u=max_u-min_u
    diff_v=max_v-min_v

    grids=[[0,0]]
    delete_num=0
    if test:
       plt.imshow(iuv)
    for i in range(grid_num):
        line=np.where(part_u==min_u+i*(diff_u//grid_num),1,0)
        for j in range(grid_num):
            point=np.where(part_v==min_v+j*(diff_v//grid_num),1,0)
            coord=np.where(point*line==1)

            if len(coord[0])>0 and ([coord[1][0]+(h//2),coord[0][0]+(w//2)]!=grids[-1]):
                if test:
                  plt.plot(coord[1][0],coord[0][0],'+')
                  plt.plot(source[i*grid_num+j-delete_num,0],source[i*grid_num+j-delete_num,1],'v')
                #grids.append([coord[1][0]/h,coord[0][0]/w])
                grids.append([coord[1][0]+(h//2), coord[0][0]+(w//2)])

            else:

                source=np.delete(source,i*grid_num+j-delete_num,0)
                delete_num+=1

    grids=grids[1:]

    if test:
       plt.show()
       plt.close()
    return source,np.array(grids)

def distoration_image(mask,part,pattern,iuv_path,test=False):
    iuv=cv2.imread(iuv_path)
    iuv=merge_body(iuv)

    if test:
      draw_new_coutour(mask,iuv)
    x,y=mask.shape

    ori_pattern=np.zeros((x*2,y*2,3),np.uint8)
    for i in range(2):
        for j in range(2):
            ori_pattern[i*x:(i+1)*x,j*y:(j+1)*y,:]=pattern[:,:,:]

    if part in top:
        seq = top_body
    else:
        seq = under_body
    body_box=[0,0,0,0]

    for p in seq:

        n_mask=np.where(iuv[:,:,0]==p,1,0)*mask

        coord = np.where((n_mask == 1))
        if len(coord[0]) < 200:
           continue

        source,box=getsource_grid(n_mask)#1
        box=list(box)

        source,dst = getdes_grid(n_mask, iuv, source,test=False)
        l,_=source.shape

        if l>=4:
          #warped = warp_image_cv(ori_pattern, (source/2.0)+0.25, (dst/2.0)+0.25, dshape=(x*2, y*2))
          n, _ = source.shape
          source = source.reshape(1, -1, 2)
          dst = dst.reshape(1, -1, 2)
          matches = []
          for i in range(1, n + 1):
              matches.append(cv2.DMatch(i, i, 0))
          tps = cv2.createThinPlateSplineShapeTransformer()

          tps.estimateTransformation(dst, source, matches)
          warped = tps.warpImage(ori_pattern)[x//2:-x//2,y//2:-y//2]
          #plt.imshow(warped)
          #plt.show()
          #plt.close()

        else:
          warped = copy.deepcopy(pattern)

        if p==2or p==7:
            body_box=box
            n_mask=mask
            r_mask=np.where(mask==1,0,1)

            for i in range(3):
              pattern[:,:,i]=warped[:,:,i]*n_mask+pattern[:,:,i]*r_mask
            #possion blending
            mask_box=get_box(n_mask)
            cen_x,cen_y=(mask_box[0]+mask_box[1])//2,(mask_box[2]+mask_box[3])//2
            x,y=n_mask.shape
            #print()
            #cv2.seamlessClone(warped,pattern,255*n_mask,(cen_y,cen_x),cv2.NORMAL_CLONE,pattern)
            #pattern=blend_image(pattern, warped, 255 * n_mask)


        else:
            out_body=out_body_part(body_box,box,x,y)

            out_body=np.where((out_body+n_mask)>0,1,0)*mask

            r_body=np.where(out_body==1,0,1)


            body_box[0]=max(box[0],body_box[0])
            body_box[1]=min(box[1],box[1])
            body_box[2]=max(body_box[2],box[2])
            body_box[3]=min(box[3],body_box[3])

            for i in range(3):
              pattern[:,:,i]=warped[:,:,i]*out_body+pattern[:,:,i]*r_body

            mask_box = get_box(out_body)
            cen_x, cen_y = int((mask_box[0] + mask_box[1]) / 2), int((mask_box[2] + mask_box[3]) / 2)
            #print(cen_x,cen_y)
            #cv2.seamlessClone(warped, pattern, n_mask, (cen_y, cen_x), cv2.NORMAL_CLONE)
            #pattern = blend_image(pattern, warped, 255 * out_body)
            #plt.imshow(np.where(out_body==1,255,0))

            #plt.imshow(warped)
            #plt.show()
            #plt.close()

    g_pattern=cv2.cvtColor(np.uint8(pattern),cv2.COLOR_RGB2GRAY)
    ori_pattern = ori_pattern[x // 2:-x // 2, y // 2:-y // 2, :]
    n_mask=np.where(g_pattern==0,1,0)

    #n_mask = cv2.medianBlur(n_mask.reshape(), 7)
    r_mask=np.where(n_mask==1,0,1)
    for i in range(3):
      pattern[:,:,i]=pattern[:,:,i]*r_mask+ori_pattern[:,:,i]*n_mask
    #pattern = blend_image(pattern, ori_pattern, 255 * n_mask)
    if test:
      plt.imshow(pattern)
      plt.show()
      plt.close()

    return pattern
#iuv=cv2.imread("IUV/output_177_IUV.png")
#iuv=cv2.imread("IUV/output_94_IUV.png")
#iuv=cv2.imread("IUV/output_129_IUV.png")
#111,273,20,171

#mask=cv2.imread("sample_mask_image/output_177_221.png",cv2.IMREAD_GRAYSCALE)
#pattern=preparepattern(mask)
#mask=np.where(mask==255,1,0)
#distoration_image(mask,169,pattern,"IUV/output_177_IUV.png",test=True)