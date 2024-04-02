import numpy as np
import cv2
import torch
import rasterio
from . import tools_bruno as tools
from . import transforms_bruno as transforms

def load_multiband(path):
    src = rasterio.open(path, "r")
    return (np.moveaxis(src.read(), 0, -1)).astype(np.uint8)


def load_grayscale(path):
    src = rasterio.open(path, "r")
    return (src.read(1)).astype(np.uint8)


class DeepGlobeDataset(torch.utils.data.Dataset):

    """
    DeepGlobe Land Cover Classification Challenge Dataset. Read images, apply 
    augmentation and preprocessing transformations.

    Args:
        fn_list (str): List containing images paths
        classes (int): list of of class-code
        img_size (int): image size
        augm (albumentations): transfromation pipeline (e.g. flip, cut, etc.)
    """

    def __init__(
        self, fn_list, classes, img_size=512, augm=None, test=False, mu=None, sig=None,
    ):
        self.img_paths = fn_list
        self.msk_paths = [x.replace("/images/", "/masks/") for x in self.img_paths]
        self.to_tensor = (
            transforms.ToTensor(classes=classes)
            if mu is None
            else transforms.ToTensorNorm(classes=classes, mu=mu, sig=sig)
        )
        self.augm = augm
        self.test = test
        self.size = img_size
        self.ncls = len(classes)

    def __getitem__(self, i):

        # read images and masks
        img = cv2.cvtColor(cv2.imread(self.img_paths[i]), cv2.COLOR_BGR2RGB)
        if self.test:
            msk = np.zeros(img.shape[:2] + (len(self.ncls),), dtype="uint8")
        else:
            map = cv2.cvtColor(cv2.imread(self.msk_paths[i]), cv2.COLOR_BGR2RGB)
            msk = tools.make_mask(
                map,
                grey_codes=tools.class_grey_deepglobe,
                rgb_codes=tools.class_rgb_deepglobe,
            )

        # apply augmentations
        data = self.to_tensor(self.augm({"image": img, "mask": msk}, self.size))
        return data["image"], data["mask"], self.img_paths[i]

    def __len__(self):
        # return length of
        return len(self.img_paths)


class LoveDADataset(torch.utils.data.Dataset):

    """
    LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation
    https://github.com/Junjue-Wang/LoveDA?ref=pythonawesome.com

    Args:
        fn_list (str): List containing images paths
        classes (int): list of of class-code
        img_size (int): image size
        augm (albumentations): transfromation pipeline (e.g. flip, cut, etc.)
    """

    def __init__(
        self, fn_list, classes, img_size=512, augm=None, test=False, mu=None, sig=None,
    ):
        self.img_paths = fn_list
        self.msk_paths = [x.replace("images_", "masks_") for x in self.img_paths]
        self.to_tensor = (
            transforms.ToTensor(classes=classes)
            if mu is None
            else transforms.ToTensorNorm(classes=classes, mu=mu, sig=sig)
        )
        self.augm = augm
        self.test = test
        self.size = img_size
        self.ncls = len(classes)

    def __getitem__(self, i):

        # read images and masks
        img = cv2.cvtColor(cv2.imread(self.img_paths[i]), cv2.COLOR_BGR2RGB)
        if self.test:
            msk = np.zeros(img.shape[:2] + (len(self.ncls),), dtype="uint8")
        else:
            msk = cv2.imread(self.msk_paths[i], cv2.IMREAD_UNCHANGED)

        # apply augmentations
        data = self.to_tensor(self.augm({"image": img, "mask": msk}, self.size))
        return {"x":data["image"], "y":data["mask"], "fn":self.img_paths[i]}

    def __len__(self):
        # return length of
        return len(self.img_paths)


class OpenEarthMapDataset(torch.utils.data.Dataset):

    """
    OpenEarthMap dataset
    Geoinformatics Unit, RIKEN AIP

    Args:
        fn_list (str): List containing images paths
        classes (int): list of of class-code
        img_size (int): image size
        augm (albumentations): transfromation pipeline (e.g. flip, cut, etc.)
    """

    def __init__(
        self, img_list, classes, img_size=512, augm=None, mu=None, sig=None,
    ):
        self.fn_imgs = [str(f) for f in img_list]
        self.fn_msks = [f.replace("/images/", "/labels/") for f in self.fn_imgs]
        self.augm = augm
        self.to_tensor = (
            transforms.ToTensor(classes=classes)
            if mu is None
            else transforms.ToTensorNorm(classes=classes, mu=mu, sig=sig)
        )
        self.size = img_size
        self.load_multiband = load_multiband
        self.load_grayscale = load_grayscale

    def __getitem__(self, idx):
        img = self.load_multiband(self.fn_imgs[idx])
        msk = self.load_grayscale(self.fn_msks[idx])

        data = self.to_tensor(self.augm({"image": img, "mask": msk}, self.size))
        return data["image"], data["mask"], self.fn_imgs[idx]

    def __len__(self):
        return len(self.fn_imgs)

