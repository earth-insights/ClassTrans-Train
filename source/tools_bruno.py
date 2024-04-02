# inspired from https://www.kaggle.com/code/burcuamirgan/deeplabv3-deepglobe-lc-classification

import numpy as np
from . import dataset

class_rgb_openearthmap = {
    "unknown": [0, 0, 0],
    "Bareland": [128, 0, 0],
    "Grass": [0, 255, 36],
    "Pavement": [148, 148, 148],
    "Road": [255, 255, 255],
    "Tree": [34, 97, 38],
    "Water": [0, 69, 255],
    "Cropland": [75, 181, 73],
    "buildings": [222, 31, 7],
}

class_grey_openearthmap = {
    "unknown": 0,
    "Bareland": 1,
    "Grass": 2,
    "Pavement": 3,
    "Road": 4,
    "Tree": 5,
    "Water": 6,
    "Cropland": 7,
    "buildings": 8,
}

class_rgb_deepglobe = {
    "unknown": [0, 0, 0],
    "urban": [0, 255, 255],
    "agriculture": [255, 255, 0],
    "rangeland": [255, 0, 255],
    "forest": [0, 255, 0],
    "water": [0, 0, 255],
    "barren": [255, 255, 255],
}

class_grey_deepglobe = {
    "unknown": 0,
    "urban": 1,
    "agriculture": 2,
    "rangeland": 3,
    "forest": 4,
    "water": 5,
    "barren": 6,
}

class_rgb_loveda = {
    "unknown": [0, 0, 0],
    "background": [250, 250, 250],
    "building": [255, 45, 30],
    "road": [255, 255, 60],
    "water": [20, 0, 250],
    "barren": [160, 130, 180],
    "forest": [35, 255, 60],
    "agriculture": [250, 200, 135],
}

class_grey_loveda = {
    "unknown": 0,
    "background": 1,
    "building": 2,
    "road": 3,
    "water": 4,
    "barren": 5,
    "forest": 6,
    "agriculture": 7,
}


def make_mask(a, grey_codes, rgb_codes):
    """
    a: semantic map (H x W x n-classes)
    """
    out = np.zeros(shape=a.shape[:2], dtype="uint8")
    for k, v in rgb_codes.items():
        mask = np.all(np.equal(a, v), axis=-1)
        out[mask] = grey_codes[k]
    return out


def make_rgb(a, grey_codes, rgb_codes):
    """
    a: labels (H x W)
    rgd_codes: dict of class-rgd code
    grey_codes: dict of label code
    """
    out = np.zeros(shape=a.shape + (3,), dtype="uint8")
    for k, v in grey_codes.items():
        out[a == v, 0] = rgb_codes[k][0]
        out[a == v, 1] = rgb_codes[k][1]
        out[a == v, 2] = rgb_codes[k][2]
    return out


def mean_var(data_gen):
    """ mean and variance computation for a generator of numpy arrays

    Mean and variance are computed in a divide and conquer fashion individally for each array.
    The results are then properly aggregated.

    Parameters
    ----------

    data_gen: generator
        data_gen is supposed to generate numpy arrays

    """

    try:
        head = next(iter(data_gen))
    except StopIteration:
        raise ValueError("You supplied an empty generator!")
    return _mean_var(*_comp(head), data_gen)


def _comp(els):
    """ individual computation for each array """
    n_el = els.size
    sum_el = els.sum()  # basically mean
    sum2_el = ((els - sum_el / n_el) ** 2).sum()  # basically variance
    return (sum_el, sum2_el, n_el)


def _mean_var(sum_a, sum2_a, n_a, data_list):
    """ divide and conquer mean and variance computation """

    def _combine_samples(sum_a, sum2_a, n_a, sum_b, sum2_b, n_b):
        """ implements formulae 1.5 in [3] """
        sum_c = sum_a + sum_b
        sum1_c = sum2_a + sum2_b
        sum1_c += ((sum_a * (n_b / n_a) - sum_b) ** 2) * (n_a / n_b) / (n_a + n_b)

        return (sum_c, sum1_c, n_a + n_b)

    for el_b in data_list:
        # iteration and aggreation
        sum_a, sum2_a, n_a = _combine_samples(sum_a, sum2_a, n_a, *_comp(el_b))
    return (sum_a / n_a, sum2_a / n_a)

