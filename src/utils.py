"""Util functions."""

import cv2
import numpy as np
# from skimage import io


def augment(imgName, angle):
    name = '../track_2_data/IMG/' + imgName.split('/')[-1]
    current_image = cv2.imread(name)
    # current_image = io.imread(name)
    current_image = current_image[65:-25, :, :]
    if np.random.rand() < 0.5:
        current_image = cv2.flip(current_image, 1)
        # current_image = np.flipud(current_image)
        angle = angle * -1.0
    return current_image, angle


def toDevice(datas, device):
    imgs, angles = datas
    return imgs.float().to(device), angles.float().to(device)
