import os
from torch.nn.modules.loss import _Loss
import torch
import torch.nn.functional as F
import random

import h5py as h5py
import numpy as np
from PIL import Image
import cv2
import matplotlib.image as imgplt
import matplotlib.pyplot as plt


def get_square(img, pos):
    """Extract a left or a right square from ndarray shape : (H, W, C))"""
    h = img.shape[0]
    if pos == 0:
        return img[:, :]
    else:
        return img[:, :]


def split_img_into_squares(img):
    return get_square(img, 0), get_square(img, 1)


def hwc_to_chw(img):
    return np.transpose(img, axes=[2, 0, 1])


def resize_and_crop(pilimg, scale=0.5, final_height=None):
    w = pilimg.size[0]
    h = pilimg.size[1]
    newW = int(w * scale)
    newH = int(h * scale)

    if not final_height:
        diff = 0
    else:
        diff = newH - final_height

    img = pilimg.resize((newW, newH))
    img = img.crop((0, diff // 2, newW, newH - diff // 2))
    return np.array(img, dtype=np.float32)


def batch(iterable, batch_size):
    """Yields lists by batch"""
    b = []
    for i, t in enumerate(iterable):
        b.append(t)
        if (i + 1) % batch_size == 0:
            yield b
            b = []

    if len(b) > 0:
        yield b


def split_train_val(dataset, val_percent=0.05):
    dataset = list(dataset)
    length = len(dataset)
    n = int(length * val_percent)
    random.shuffle(dataset)
    return {'train': dataset[:-n], 'val': dataset[-n:]}


def normalize(x):
    return x / 255


def merge_masks(img1, img2, full_w):
    h = img1.shape[0]

    new = np.zeros((h, full_w), np.float32)
    new[:, :full_w // 2 + 1] = img1[:, :full_w // 2 + 1]
    new[:, full_w // 2 + 1:] = img2[:, -(full_w // 2 - 1):]

    return new


# credits to https://stackoverflow.com/users/6076729/manuel-lagunas
def rle_encode(mask_image):
    pixels = mask_image.flatten()
    # We avoid issues with '1' at the start or end (at the corners of
    # the original image) by setting those pixels to '0' explicitly.
    # We do not expect these to be non-zero for an accurate mask,
    # so this should not harm the score.
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] = runs[1::2] - runs[:-1:2]
    return runs


def ImageToMatrix():
    # 读取图片
    path = '/home/panmeng/data/nyu_depths/4.png'
    # path = '/home/panmeng/data/sceneflow/driving__disparity/15mm_focallength/scene_backwards/fast/left/0001.pfm'
    # im = Image.open(path)
    # image_arr = np.array(im)
    # print(image_arr)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    print("图像的形状,返回一个图像的(行数,列数,通道数):", img.shape)
    print("图像的像素数目:", img.size)
    print("图像的数据类型:", img.dtype)
    image_arr = np.array(img)
    print(image_arr)
    with open('pixel.txt', 'w') as f:
        f.write(str(image_arr))
    # img = imgplt.imread(path)
    # plt.imshow(img)
    # plt.savefig('grayimg')
    # plt.show()


def extractNYU(path):
    f = h5py.File(path)
    print(f, type(f))
    images = f['images']
    print(images, type(images))
    images = np.array(images)

    path_converted = '/home/panmeng/data/'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    images_number = []
    for i in range(len(images)):
        print(i)
        images_number.append(images[i])
        a = np.array(images_number[i])
        #    print len(img)
        # img=img.reshape(3,480,640)
        #   print img.shape
        r = Image.fromarray(a[0]).convert('L')
        g = Image.fromarray(a[1]).convert('L')
        b = Image.fromarray(a[2]).convert('L')
        img = Image.merge("RGB", (r, g, b))
        img = img.transpose(Image.ROTATE_270)
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        # os.mkdir('/home/panmeng/data/nyu_images')
        iconpath = '/home/panmeng/data/nyu_images/' + str(i) + '.png'
        img.save(iconpath, optimize=True)


def extract_depths_NYU(path):
    f = h5py.File(path)
    print(f, type(f))
    depths = f['depths']
    depths = np.array(depths)
    path_converted = '/home/panmeng/data/nyu_depths/'
    if not os.path.isdir(path_converted):
        os.makedirs(path_converted)

    max = depths.max()
    print(depths.shape)
    print(depths.max())
    print(depths.min())

    depths = depths / max * 255
    depths = depths.transpose((0, 2, 1))
    print(depths.max())
    print(depths.min())

    for i in range(len(depths)):
        print(str(i) + '.png')
        depths_img = Image.fromarray(np.uint8(depths[i]))
        depths_img = depths_img.transpose(Image.FLIP_LEFT_RIGHT)

        iconpath = path_converted + str(i) + '.png'
        depths_img.save(iconpath, 'PNG', optimize=True)
    pass


def readimg():
    path1 = '/data/sync/basement_0001a/sync_depth_00001.png'
    path2 = '/home/panmeng/data/nyu_depths/train_dir/10.png'
    img = Image.open(path1)
    #img = img.convert('L')
    img_matrix = np.array(img)
    pass

#def label_smooth(target, classes, epsilon):
#    one_hot = F.one_hot(target,classes)
#    smooth_label = one_hot*(1 - epsilon) + torch.ones_like(one_hot)*epsilon
#    smooth_label = smooth_label.transpose(0,2)
#    return smooth_label
#
#class label_smooth_crossentropy(_Loss):
#    def __init__(self, size_average=None, reduce=None, reduction='mean'):
#        super(label_smooth_crossentropy, self).__init__(size_average)
#    def forward(self,input,target):
#        return torch.sum(-(torch.log_softmax(input,1).matmul(target)))

def label_smooth(target, classes, epsilon):
    one_hot = F.one_hot(target,classes)
    smooth_label = one_hot*(1 - epsilon) + torch.ones_like(one_hot)*epsilon
    smooth_label = smooth_label.transpose(0,2)
    return smooth_label

class label_smooth_crossentropy(_Loss):
    def __init__(self, size_average=None, reduce=None, reduction='mean'):
        super(label_smooth_crossentropy, self).__init__(size_average)
    def forward(self,input,target):
        return torch.sum(-(torch.log_softmax(input,1).matmul(target)))

if __name__ == '__main__':
    readimg()
