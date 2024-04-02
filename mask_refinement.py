import numpy as np
import glob
import cv2
import os
from PIL import Image
import segmentation_refinement as refine

label_pths = glob.glob('/home/zbh/lrx/keyan3/CVPR2024-OEM-Fewshot-curr-best/results/pred_softmax'+'/*.npy')
label_pths_png=glob.glob('/home/zbh/lrx/keyan3/CVPR2024-OEM-Fewshot-curr-best/results/preds'+'/*.png')
OUT_DIR_building_2 = '/home/zbh/lrx/keyan3/CVPR2024-OEM-Fewshot-curr-best/results/building_type_2'
OUT_DIR_building_1 = '/home/zbh/lrx/keyan3/CVPR2024-OEM-Fewshot-curr-best/results/building_type_1'
label_paths_tif='/home/zbh/lrx/keyan3/data_new/queryset/images'

refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

for fn_img in label_pths_png:
    mask=cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)  
    filename = os.path.splitext(os.path.basename(fn_img))[0]
    image=cv2.imread(os.path.join(label_paths_tif, filename+'.tif'))
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    mash_num=11
    mask[mask != mash_num] = 0
    mask[mask == mash_num] = 255
    output = refiner.refine(image, mask, fast=False, L=1000) 
    a=20
    output[output >= a]=255
    output[output < a]=0
    output[output == 255]=mash_num
    print(output.shape)
    # this line to save output
    cv2.imwrite(os.path.join(
                    OUT_DIR_building_2, filename + '.png'), output)
print("building type 2 processing finished")

for fn_img in label_pths_png:

    mask=cv2.imread(fn_img, cv2.IMREAD_GRAYSCALE)  
    filename = os.path.splitext(os.path.basename(fn_img))[0]
    image=cv2.imread(os.path.join(label_paths_tif, filename+'.tif'))
    image = cv2.resize(image, (1024, 1024), interpolation=cv2.INTER_NEAREST)
    mash_num=7
    mask[mask != mash_num] = 0
    mask[mask == mash_num] = 255
    output = refiner.refine(image, mask, fast=False, L=1000) 
    a=20
    output[output >= a]=255
    output[output < a]=0
    output[output == 255]=mash_num
    cv2.imwrite(os.path.join(
                    OUT_DIR_building_1, filename + '.png'), output)
print("building type 1 processing finished")
