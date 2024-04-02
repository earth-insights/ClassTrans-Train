"""
    .npy文件放置在 ``data/cvpr2024_OEM/valset/real_npy`` 下
    将.npy转换为.tif
"""

import os
import os.path as osp

import cv2
import numpy as np

if __name__ == '__main__':
    src_dir = 'data/cvpr2024_OEM/valset/real_npy'
    dst_dir = 'data/cvpr2024_OEM/valset/real'
    dst_txt_file = 'data/cvpr2024_OEM/val.txt'

    os.makedirs(dst_dir, exist_ok=True)

    file_list = []
    for file_name in os.listdir(src_dir):
        img = np.load(osp.join(src_dir, file_name))
        img = img.astype(np.uint8)

        dst_file_name = file_name.split('.')[0]+'.tif'
        file_list.append(dst_file_name)
        cv2.imwrite(osp.join(dst_dir, dst_file_name), img)

    with open(dst_txt_file, 'w') as f:
        for item in file_list:
            f.write(item + '\n')

    