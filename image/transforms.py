import torch
import math

from tools.image import cv
from tools import Struct

import random

default_statistics = Struct(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

def normalize_batch(batch, mean=default_statistics.mean, std=default_statistics.mean):
    assert(batch.size(3) == 3)
    batch = batch.float().div_(255)

    for i in range(0, 3):
        batch.select(3, i).sub_(mean[i]).div_(std[i])

    return batch.permute(0, 3, 1, 2)

def un_normalize_batch(batch, mean=default_statistics.mean, std=default_statistics.mean):
    assert(batch.size(1) == 3)
    batch = batch.clone()

    for i in range(0, 3):
        batch.select(1, i).mul_(std[i]).add_(mean[i])

    batch = batch.mul_(255).clamp(0, 255).byte()
    return batch.permute(0, 2, 3, 1)


def random_gamma(image, gamma_range = 0.1):
    gamma = random.uniform(1-gamma_range, 1+gamma_range)
    return cv.adjust_gamma(image, gamma)


def adjust_gamma(adjustment=0.1, per_channel=0):
    def f(image):
        if per_channel > 0:
            for d in range(0, 3):
                image.select(2, d).copy_(random_gamma(image.select(2, d), per_channel))

        if adjustment > 0:
            image = random_gamma(image, adjustment)

        return image
    return f


def scaling(sx, sy):
    return torch.DoubleTensor ([
      [sx, 0, 0],
      [0, sy, 0],
      [0, 0, 1]])

def rotation(a):
    sa = math.sin(a)
    ca = math.cos(a)

    return torch.DoubleTensor ([
      [ca, -sa, 0],
      [sa,  ca, 0],
      [0,   0, 1]])

def translation(tx, ty):
    return torch.DoubleTensor ([
      [1, 0, tx],
      [0, 1, ty],
      [0, 0, 1]])


def random_check(lower, upper):
    if (lower >= upper):
        return (lower + upper) / 2
    else:
        return random.randint(lower, upper)

def random_region(image_size, size, border = 0):

    w, h = image_size[1], image_size[0]
    tw, th = size

    x1 = random_check(border, w - tw - border)
    y1 = random_check(border, h - th - border)

    return (x1, y1), (x1 + tw, y1 + th)

def random_crop(dim, border=0):
    def crop(image):
        h, w, c = image.size()
        assert dim[0] + border <= w and dim[1] + border <= h

        pos, _ = random_region(image.size(), dim, border)
        return image.narrow(0, pos[1], dim[1]).narrow(1, pos[0], dim[0])
    return crop

def centre_crop(dim):
    def crop(image):
        h, w, c = image.size()
        assert dim[0] <= w and dim[1]  <= h

        pos = ((w - dim[0]) // 2, (h - dim[1]) // 2)

        return image.narrow(0, pos[1], dim[1]).narrow(1, pos[0], dim[0])
    return crop


def clamp(lower, upper, *xs):
    return min(upper, max(lower, *xs))

def randoms(*ranges):
    pair = lambda r: (r, -r) if isinstance(r, int) else r
    return [random.uniform(*pair(r)) for r in ranges]


def compose(*functions):
    def composed(image):
        for f in functions:
            image = f(image)
        return image
    return composed


border = cv.border

def warp_affine(image, t, dest_size, border_mode=border.constant, border_fill=default_statistics.mean):
    border_fill  = [255 * x for x in border_fill]
    return cv.warpAffine(image, t, dest_size, flags = cv.inter.cubic, borderMode=border_mode, borderValue = border_fill)

def warp_perspective(image, t, dest_size, border_mode=border.constant, border_fill=default_statistics.mean):
    border_fill  = [255 * x for x in border_fill]
    return cv.warpPerspective(image, t, dest_size, flags = cv.inter.cubic, borderMode=border_mode, borderValue = border_fill)


def affine_crop(input_crop, dest_size, scale_range=(1, 1), rotation_size=0, border_mode=border.constant, border_fill=default_statistics.mean):
    def f(image):
        t = make_affine_crop(image.size(), input_crop, dest_size, scale_range, rotation_size)
        return warp_affine(image, t, dest_size, border_fill)
    return f


def affine_crop(input_crop, dest_size, scale_range=(1, 1), rotation_size=0, border_mode=border.constant, border_fill=default_statistics.mean):
    def f(image):
        t = make_affine_crop(image.size(), input_crop, dest_size, scale_range, rotation_size)
        return warp_affine(image, t, dest_size, border_mode=border_mode, border_fill=border_fill)
    return f


def image_augmentation(dest_size, affine_jitter=0, perspective_jitter=0, translation=0, scale_range=(1, 1), rotation_size=0, flip=False, border_mode=border.constant, border_fill=default_statistics.mean):
    def f(image):
        t = make_affine_augmentation(image.size(), dest_size, translation=translation, flip=flip, scale_range=scale_range, rotation_size=rotation_size)

        if affine_jitter > 0:
            t = make_affine_jitter(dest_size, affine_jitter).mm(t)

        if perspective_jitter > 0:
            t = make_perspective_jitter(dest_size, perspective_jitter).mm(t)

        return warp_perspective(image, t, dest_size, border_mode=border_mode, border_fill=border_fill)
    return f


def make_perspective_jitter(image_size, pixels=1):
    (w, h) = image_size

    corners = torch.FloatTensor ([(0, 0), (0, h), (w, h), (w, 0)])
    dest = torch.FloatTensor(4, 2).uniform_(-pixels, pixels) + corners

    return cv.getPerspectiveTransform(corners, dest)

def make_affine_jitter(image_size, pixels=1):
    (w, h) = image_size

    corners = torch.FloatTensor ([(0, 0), (0, h), (w, h)])
    dest = torch.FloatTensor(3, 2).uniform_(-pixels, pixels) + corners

    t = torch.eye(3).double()
    t.narrow(0, 0, 2).copy_(cv.getAffineTransform(corners, dest))
    return t


def perspective_jitter(pixels=1, border_mode=border.constant, border_fill=default_statistics.mean):
    def f(image):
        size = (image.size(1), image.size(0))
        t = make_perspective_jitter(size, pixels=pixels)
        return warp_perspective(image, t, size, border_mode=border_mode, border_fill=border_fill)
    return f



def make_affine(dest_size, centre, scale=1, rot=0, flip=1):
    sx, sy = scale

    toCentre = translation(-centre[0], -centre[1])
    fromCentre = translation(dest_size[0] * 0.5, dest_size[1] * 0.5)

    r = rotation(rot * (math.pi / 180))
    s = scaling(flip * sx, 1 * sy)
    return fromCentre.mm(s).mm(r).mm(toCentre)


def make_affine_augmentation(image_size, dest_size, translation=0, scale_range=(1, 1), rotation_size=0, flip=False):
    """ Applying affine augmentation to an image, random translation, rotation and scale """

    tx, ty, s, r, flip = randoms(translation, translation, scale_range, rotation_size, 1 if flip else 0)
    centre = (image_size[0] * 0.5 + tx, image_size[1] * 0.5 + ty)

    return make_affine(dest_size, centre, (s, s), r, 1 if flip > 0 else -1)



def make_affine_crop(image_size, input_crop, dest_size, scale_range=(1, 1), rotation_size=0, border=0):
    """ Cropping a smaller region out of a larger image with affine augmentation """

    min_scale = clamp(scale_range[0], scale_range[1], dest_size[0] / image_size[1], dest_size[1] / image_size[0])
    scale = random.uniform(min_scale, scale_range[1])

    crop_size = (math.floor(1/scale * input_crop[0]), math.floor(1/scale * input_crop[1]))
    centre, extents = random_region(image_size, crop_size, border)

    rotation = random.uniform(-rotation_size, rotation_size)
    return make_affine(dest_size, centre, (scale, scale), rotation)
