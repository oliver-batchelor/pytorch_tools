
import sys
import cv2
import torch

import numpy as np
from tools import Struct

default_statistics = Struct(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])



def _bgr_rgb(cv_image):
    if(len(cv_image.shape) == 3 and cv_image.shape[2] == 3):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    return cv_image

def imread(path, flag=cv2.IMREAD_UNCHANGED):
    cv_image = _bgr_rgb(cv2.imread(path, flag))
    assert cv_image is not None, "imread: failed to load " + pathf

    image = torch.from_numpy (cv_image)

    if(image.dim() == 2):
        image = image.view(*image.size(), 1)

    return image

def imread_color(path):
    image = imread(path, cv2.IMREAD_COLOR)
    assert image.size(2) >= 3
    return image.narrow(2, 0, 3)

def imread_gray(path):
    return imread(path, cv2.IMREAD_GRAYSCALE)


def write(image, extension, path):
    result, buf = imencode(extension, image)
    with open(path, 'wb') as file:
        file.write(buf)

def imencode(extension, image):
    assert(image.dim() == 3 and image.size(2) <= 4)
    return cv2.imencode(extension, _bgr_rgb(image.numpy()))

def imwrite(path, image):
    assert(image.dim() == 3 and image.size(2) <= 4)
    return cv2.imwrite(path, _bgr_rgb(image.numpy()))

waitKey = cv2.waitKey

def display(t):
    imshow("image", t)
    return waitKey()

def imshow(name, t):
    cv2.imshow(name, _bgr_rgb(t.numpy()))
    waitKey(1)


def display_bgr(t):
    imshow_bgr("image", t)
    return waitKey()

def imshow_bgr(name, t):
    cv2.imshow(name, t.numpy())
    waitKey(1)



def adjust_gamma(image, gamma=1.0):
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")

	return torch.from_numpy(cv2.LUT(image.numpy(), table))

def cvtColor(image, conversion):
    return torch.from_numpy(cv2.cvtColor(image.numpy(), conversion))

def bgr_to_hsv(image):
    return cvtColor(image, cv2.COLOR_BGR2HSV)

def hsv_to_bgr(image):
    return cvtColor(image, cv2.COLOR_HSV2BGR)

def bgr_to_rgb(image):
    return cvtColor(image, cv2.COLOR_BGR2RGB)

def rgb_to_bgr(image):
    return cvtColor(image, cv2.COLOR_RGB2BGR)

def warpAffine(image, t, target_size, **kwargs):
    t = t.narrow(0, 0, 2)
    return torch.from_numpy(cv2.warpAffine(image.numpy(), t.numpy(), target_size, **kwargs))


def getPerspectiveTransform(source_points, dest_points):
    assert source_points.size(0) == 4 and source_points.size(1) == 2
    dest_points.size(0) == 4 and source_points.size(1) == 2
    return torch.from_numpy(cv2.getPerspectiveTransform(source_points.numpy(), dest_points.numpy()))

def getAffineTransform(source_points, dest_points):
    assert source_points.size(0) == 3 and source_points.size(1) == 2
    dest_points.size(0) == 3 and source_points.size(1) == 2
    return torch.from_numpy(cv2.getAffineTransform(source_points.numpy(), dest_points.numpy()))


def warpPerspective(image, t, target_size, **kwargs):
    return torch.from_numpy(cv2.warpPerspective(image.numpy(), t.numpy(), target_size, **kwargs))

def resize(image, dim, **kwargs):
    channels = image.size(2)
    result = torch.from_numpy(cv2.resize(image.numpy(), dim, **kwargs))
    return result.view(dim[1], dim[0], channels)

inter = Struct(cubic = cv2.INTER_CUBIC, nearest = cv2.INTER_NEAREST, area = cv2.INTER_AREA)

border = Struct(
    replicate=cv2.BORDER_REPLICATE,
    wrap=cv2.BORDER_REPLICATE,
    constant=cv2.BORDER_CONSTANT,
    reflect=cv2.BORDER_REFLECT
)

image_read = Struct(
    unchanged=cv2.IMREAD_UNCHANGED,
    color=cv2.IMREAD_COLOR,
    greyscale=cv2.IMREAD_GRAYSCALE
)
