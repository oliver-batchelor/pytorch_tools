
from tools import struct

import sys
import cv2
import torch

import numpy as np
from tools import struct



def _rgb_bgr(cv_image):
    if(len(cv_image.shape) == 3 and cv_image.shape[2] == 3):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    return cv_image

def _bgr_rgb(cv_image):
    if(len(cv_image.shape) == 3 and cv_image.shape[2] == 3):
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    return cv_image

def convert_loaded(cv_image):
    cv_image = _bgr_rgb(cv_image)
    image = torch.from_numpy (cv_image)

    if(image.dim() == 2):
        image = image.view(*image.size(), 1)
    return image


def video_capture(path):
    cap = cv2.VideoCapture(path)

    def frames(start = 0):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                yield convert_loaded(frame)
            else:
                raise StopIteration


        raise StopIteration


    if cap.isOpened():
        return frames, struct(
            fps = cap.get(cv2.CAP_PROP_FPS),
            size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

    assert False, "video_capture: failed to load " + str(path)



def imread_depth(path):
    cv_image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    if cv_image.dtype == np.uint16:
        cv_image = cv_image.astype(np.float32)
    else:
        assert False, "imread_depth - unsupported type {}: {}".format(str(cv_image.dtype), path)
    
    image = torch.from_numpy (cv_image)
    if(image.dim() == 2):
        image = image.view(*image.size(), 1)

    return image

def imread(path, flag=cv2.IMREAD_UNCHANGED):
    cv_image = cv2.imread(path, flag)
    assert cv_image is not None, "imread: failed to load " + str(path)

    return convert_loaded(cv_image)

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

def display(t, name="image"):
    cv2.namedWindow(name)

    imshow(t, name=name)

    while cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) > 0:
        keyCode = waitKey(10)
        if keyCode >= 0:
            return keyCode

    return -1

def imshow(t, name="image"):
    cv2.imshow(name, _rgb_bgr(t.numpy()))
    waitKey(1)


# def display_bgr(t):
#     imshow_bgr("image", t)
#     return waitKey()

# def imshow_bgr(name, t):
#     cv2.imshow(name, t.numpy())
#     waitKey(1)





def multiply_add(image, a, b):
    i = image.numpy()
    return torch.from_numpy(cv2.addWeighted(i, a, i, 0, b))

def add(image, b):
    return torch.from_numpy(cv2.add(image.numpy(), b))


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


def rgb_to_hsv(image):
    return cvtColor(image, cv2.COLOR_RGB2HSV)

def hsv_to_rgb(image):
    return cvtColor(image, cv2.COLOR_HSV2RGB)


def rgb_to_gray(image):
    return cvtColor(image, cv2.COLOR_RGB2GRAY)

def gray_to_rgb(image):
    return cvtColor(image, cv2.COLOR_GRAY2RGB)


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

def flip_horizontal(image):
    return torch.from_numpy(cv2.flip(image.numpy(), 1))

def flip_vertical(image):
    return torch.from_numpy(cv2.flip(image.numpy(), 0))


def int_list(p):
    if type(p) is torch.Tensor:
        return tuple(p.int().tolist())
    else:
        return tuple(p)


line_type = struct (
    filled = cv2.FILLED,
    line4 = cv2.LINE_4,
    line8 = cv2.LINE_8,
    lineAA = cv2.LINE_AA
)


def blend_over(dest, src):
    dh, dw, dc = dest.shape
    sh, sw, sc = src.shape

    assert dc == 3 and sc == 4

    if [sh, sw] != [dh, dw]:
        src = cv.resize(src, (dw, dh))

    alpha = src.select(2, sc - 1)
    color = src.narrow(2, sc - 1, 1)

    return dest * (1 - alpha) + color * alpha  


def rectangle(image, lower, upper, color=(255, 255, 255, 255), thickness=1, line=line_type.lineAA):
    assert image.is_contiguous()

    image = image.numpy()
    cv2.rectangle(image, int_list(lower), int_list(upper), color=int_list(color), thickness=int(thickness), lineType = line)

    return torch.from_numpy(image)


    # cv2.putText(img,'OpenCV',(10,500), font, 4,(255,255,255),2,cv2.LINE_AA)


def putText(image, text, pos, scale=1, color=(255, 255, 255, 255), thickness=1, line=line_type.lineAA):
    assert image.is_contiguous()

    image = image.numpy()

    cv2.putText(image, str(text), int_list(pos), fontFace=1, fontScale=scale, color=int_list(color), thickness=int(thickness), lineType = line)

    return torch.from_numpy(image)


inter = struct(cubic = cv2.INTER_CUBIC, nearest = cv2.INTER_NEAREST, area = cv2.INTER_AREA)

border = struct(
    replicate=cv2.BORDER_REPLICATE,
    wrap=cv2.BORDER_REPLICATE,
    constant=cv2.BORDER_CONSTANT,
    reflect=cv2.BORDER_REFLECT
)

image_read = struct(
    unchanged=cv2.IMREAD_UNCHANGED,
    color=cv2.IMREAD_COLOR,
    greyscale=cv2.IMREAD_GRAYSCALE
)
