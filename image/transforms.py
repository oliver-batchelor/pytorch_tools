import torch
import math
import numbers

from tools.image import cv
from tools import struct

import random

default_statistics = struct(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def normalize_batch(batch, mean=default_statistics.mean, std=default_statistics.std):
    assert(batch.size(3) == 3)
    batch = batch.float().div_(255)

    for i in range(0, 3):
        batch.select(3, i).sub_(mean[i]).div_(std[i])

    return batch.permute(0, 3, 1, 2)



def un_normalize_batch(batch, mean=default_statistics.mean, std=default_statistics.std):
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

def adjust_brightness(brightness = 0, contrast = 0):
    def f(image):
        b = random.uniform(-brightness, brightness) * 255
        c = random.uniform(1 - contrast, 1 + contrast)

        image = image.float().add_(b).mul_(c).clamp_(0, 255).byte()
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


def random_crop(image_size, crop_size, border_bias = 0):
    width, height = image_size
    crop_width, crop_height = crop_size

    def random_interval(lower, upper, size, border_bias = 0):
        max_size = upper - lower

        if size < max_size:
            pos = random.uniform(lower - border_bias, upper - size + border_bias)
            pos = clamp(lower, upper - size)

            return pos, size
        else:
            return lower + border, max_size    

    x, w = random_interval(0, width, crop_width)
    y, h = random_interval(0, height, crop_height)

    return (x, y), (w, h)



def random_crop_padded(image_size, crop_size, border_bias = 0):

    def random_offset(lower, upper, size):
        if size > upper - lower:
            return random.uniform(upper - size, lower)
        else:
            pos = random.uniform(lower - border_bias, upper - size + border_bias)
            return clamp(lower, upper - size, pos)

    width, height = image_size
    crop_width, crop_height = crop_size

    x = random_offset(0, width, crop_width)
    y = random_offset(0, height, crop_height)

    return x, y


def clamp(lower, upper, *xs):
    return min(upper, max(lower, *xs))

def randoms(*ranges):
    pair = lambda r: (r, -r) if isinstance(r, numbers.Number) else r
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
    return cv.warpAffine(image, t, dest_size, flags = cv.inter.area, borderMode=border_mode, borderValue = border_fill)


def warp_affine(image, t, dest_size, border_mode=border.constant, border_fill=default_statistics.mean):
    border_fill  = [255 * x for x in border_fill]
    return cv.warpAffine(image, t, dest_size, flags = cv.inter.area, borderMode=border_mode, borderValue = border_fill)


def resize_to(image, dest_size):
    return cv.resize(image, dest_size, interpolation = cv.inter.area)



def resize_scale(image, scale):
    input_size = (image.size(1), image.size(0))
    dest_size = (int(input_size[0] * scale), int(input_size[1] * scale))
    return resize_to(image, dest_size)

def adjust_scale(scale):
    return lambda image: resize_scale(image, scale)


def warp_perspective(image, t, dest_size, border_mode=border.constant, border_fill=default_statistics.mean):
    border_fill  = [255 * x for x in border_fill]
    return cv.warpPerspective(image, t, dest_size, flags = cv.inter.area, borderMode=border_mode, borderValue = border_fill)


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

def centre_on(image, dest_size, border_mode=border.constant, border_fill=(0, 0, 0)):
    h, w, c = image.size()
    
    t = translation((w - dim[0]) / 2, (h - dim[1]) / 2)
    return warp_affine(image, t, dest_size, border_mode=border_mode, border_fill=border_fill)


def image_augmentation(dest_size, affine_jitter=0, perspective_jitter=0, translation=0, scale_range=(1, 1), rotation_size=0, flip=True, border_mode=border.constant, border_fill=default_statistics.mean):
    def f(image):
        t = random_affine(image.size(), dest_size, translation=translation, flip=flip, scale_range=scale_range, rotation_size=rotation_size)

        if affine_jitter > 0:
            t = random_affine_jitter(dest_size, affine_jitter).mm(t)

        if perspective_jitter > 0:
            t = random_perspective_jitter(dest_size, perspective_jitter).mm(t)

        return warp_perspective(image, t, dest_size, border_mode=border_mode, border_fill=border_fill)
    return f


def random_perspective_jitter(image_size, pixels=1):
    (w, h) = image_size

    corners = torch.FloatTensor ([(0, 0), (0, h), (w, h), (w, 0)])
    dest = torch.FloatTensor(4, 2).uniform_(-pixels, pixels) + corners

    return cv.getPerspectiveTransform(corners, dest)


def random_affine_jitter(image_size, pixels=1):
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



def make_affine(dest_size, centre, scale=(1, 1), rot=0, flip=1):
    sx, sy = scale

    toCentre = translation(-centre[0], -centre[1])
    fromCentre = translation(dest_size[0] * 0.5, dest_size[1] * 0.5)

    r = rotation(rot * (math.pi / 180))
    s = scaling(flip * sx, 1 * sy)
    return fromCentre.mm(s).mm(r).mm(toCentre)


def random_affine(image_size, dest_size, translation=0, scale_range=(1, 1), rotation_size=0, flip=False):
    """ Applying affine augmentation to an image, random translation, rotation and scale """

    tx, ty, s, r, flip = randoms(translation, translation, scale_range, rotation_size, 1 if flip else 0)
    centre = (image_size[0] * 0.5 + tx, image_size[1] * 0.5 + ty)

    return make_affine(dest_size, centre, (s, s), r, 1 if flip < 0.5 else -1)


def perspective_transform(t, points):
    points = points.mm(t.t().float())

    w =  points.narrow(1, 2, 1)
    points = points.narrow(1, 0, 2)
    return points / w.expand_as(points)

def fit_transform(input_size, t, pad=0):
    (w, h) = input_size

    input_corners = torch.FloatTensor ([(-pad, -pad, 1), (-pad, h+pad, 1), (w+pad, h+pad, 1), (w+pad, -pad, 1)])
    corners = perspective_transform(t, input_corners)

    (l, _), (u, _) = corners.min(0), corners.max(0)
    dest_size = u - l
    offset =  -l

    return translation(*offset).mm(t), tuple(dest_size.long())
