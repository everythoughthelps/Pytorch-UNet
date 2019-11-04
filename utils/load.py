#
# load.py : utils on generators / lists of ids to transform from strings to
#           cropped images and masks

import os

import numpy as np
from PIL import Image

from .utils import resize_and_crop, get_square, normalize, hwc_to_chw


def get_ids(dir):
    """Returns a list of the ids in the directory"""
    print(os.listdir(dir))
    print(len(os.listdir(dir)))
    return (f[:-4] for f in os.listdir(dir))


def split_ids(ids, n=2):
    """Split each id in n, creating n tuples (id, k) for each id"""
    return ((id, i)  for id in ids for i in range(n))


def to_cropped_imgs(ids, dir, suffix, scale):
    """From a list of tuples, returns the correct cropped img"""
    for id in ids:
        im = resize_and_crop(Image.open(dir + id + suffix), scale=scale)
        yield im

def get_imgs_and_masks(ids, dir_img, dir_mask, scale):
    """Return all the couples (img, mask)"""
    imgs = to_cropped_imgs(ids, dir_img, '.png', scale)
    # need to transform from HWC to CHW
    imgs_switched = map(hwc_to_chw, imgs)
    #imgs_normalized = map(normalize, imgs_switched)

    masks = to_cropped_imgs(ids, dir_mask, '.png', scale)

    return zip(imgs_switched, masks)


def get_full_img_and_mask(ids,dir_img, dir_mask):
    imgs_list = []
    mask_list =[]
    for id in ids:
        imgs = Image.open(dir_img + id + '.png')
        imgs = hwc_to_chw(imgs)
        imgs = np.array(imgs)
        imgs_list.append(imgs)
        mask = Image.open(dir_mask + id + '.png')
        mask = np.array(mask)
        mask_list.append(mask)
    return imgs_list,mask_list