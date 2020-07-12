import torch
import torch.nn as nn
import torch.nn.functional as F

import cv2 as cv
import numpy as np

from PIL import Image

import random, math


def resize_images(images, H, W):
    """
    This function resizes images to fit the network
    """
    new_images = []

    N, C, H0, W0 = images.shape

    for i in range(images.shape[0]):
        img = images[i].numpy()  # [C,H,W]

        img = np.transpose(img, (1, 2, 0))  # [H,W,C]

        d = random.randint(0, 360 - 1)

        """A bilinear interpolation is used by default"""
        new_img = cv.resize(img, dsize=(H, W)).reshape(H, W, C)

        new_img = np.transpose(new_img, (2, 0, 1))  # [C,H,W]
        new_images.append(new_img)

    new_images = torch.from_numpy(np.stack(new_images, axis=0))

    return new_images


def random_rotate(images, dataset='cifar10'):
    """
    This function takes multiple images, and rotate each image by a random angle [0, 360)
    """
    new_images = []

    (N, C, H, W) = images.shape

    for i in range(images.shape[0]):
        img = images[i].numpy()  # [C,H,W]

        img = np.transpose(img, (1, 2, 0))  # [H,W,C]

        d = random.randint(0, 360 - 1)

        M = cv.getRotationMatrix2D((H // 2, W // 2), d, 1.0)
        img_rot = cv.warpAffine(img, M, (H, W)).reshape(H, W, C)

        img_rot = np.transpose(img_rot, (2, 0, 1))  # [C,H,W]
        new_images.append(img_rot)

    new_images = torch.from_numpy(np.stack(new_images, axis=0))

    return new_images


def random_translate(images, dataset='cifar10'):
    """
    This function takes multiple images, and translates each image randomly by at most quarter of the image.
    """

    (N, C, H, W) = images.shape

    min_pixel = torch.min(images).item()

    new_images = []
    for i in range(images.shape[0]):
        img = images[i].numpy()  # [C,H,W]
        img = np.transpose(img, (1, 2, 0))  # [H,W,C]

        dx = random.randrange(-8, 9, 1)
        dy = random.randrange(-8, 9, 1)

        M = np.float32([[1, 0, dx], [0, 1, dy]])
        image_trans = cv.warpAffine(img, M, (H, W)).reshape(H, W, C)

        image_trans = np.transpose(image_trans, (2, 0, 1))  # [C,H,W]
        new_images.append(image_trans)

    new_images = torch.tensor(np.stack(new_images, axis=0), dtype=torch.float32)

    return new_images


def polar_transform(images, transform_type='linearpolar'):
    """
    This function takes multiple images, and apply polar coordinate conversion to it.
    """
    
    (N, C, H, W) = images.shape

    for i in range(images.shape[0]):

        img = images[i].numpy()  # [C,H,W]
        img = np.transpose(img, (1, 2, 0))  # [H,W,C]

        if transform_type == 'logpolar':
            img = cv.logPolar(img, (H // 2, W // 2), W / math.log(W / 2), cv.WARP_FILL_OUTLIERS).reshape(H, W, C)
        elif transform_type == 'linearpolar':
            img = cv.linearPolar(img, (H // 2, W // 2), W / 2, cv.WARP_FILL_OUTLIERS).reshape(H, W, C)
        img = np.transpose(img, (2, 0, 1))

        images[i] = torch.from_numpy(img)

    return images
