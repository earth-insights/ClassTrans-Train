import sys
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import source
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, FocalLoss
import glob
import torchvision.transforms.functional as TF
import math
import cv2
from PIL import Image
import time
import warnings
from pathlib import Path


class_rgb = {
    "bg": [0, 0, 0],
    "tree": [34, 97, 38],
    "rangeland": [0, 255, 36],
    "bareland": [128, 0, 0],
    "agric land type 1": [75, 181, 73],
    "road type 1": [255, 255, 255],
    "sea, lake, & pond": [0, 69, 255],
    "building type 1": [222, 31, 7],
}

class_gray = {
    "bg": 0,
    "tree": 1,
    "rangeland": 2,
    "bareland": 3,
    "agric land type 1": 4,
    "road type 1": 5,
    "sea, lake, & pond": 6,
    "building type 1": 7,
}

def label2rgb(a):
    """
    a: labels (HxW)
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in class_gray.items():
        out[a == v, 0] = class_rgb[k][0]
        out[a == v, 1] = class_rgb[k][1]
        out[a == v, 2] = class_rgb[k][2]
    return out


label_pths = glob.glob('data/cvpr2024_OEM/trainset/labels'+'/*.tif')
OUT_DIR = 'data/cvpr2024_OEM/trainset/labels_vis'
os.makedirs(OUT_DIR, exist_ok=True)

for fn_img in label_pths:
  img = cv2.imread(fn_img, -1)
  img_rgb = label2rgb(img)
  filename = os.path.splitext(os.path.basename(fn_img))[0]
  Image.fromarray(img_rgb).save(os.path.join(OUT_DIR, filename+'.png'))