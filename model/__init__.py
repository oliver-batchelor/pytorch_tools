import torch
import torch.nn.functional as F

def match_size_2d(t, sized):
    assert t.dim() == 4 and sized.dim() == 4
    dh = sized.size(2) - t.size(2)
    dw = sized.size(3) - t.size(3)

    pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    return F.pad(t, pad)


def centre_crop(t, size):
    dw = size[3] - t.size(3)
    dh = size[2] - t.size(2)

    padding = (dw//2, dw - dw//2, dh//2, dh - dh//2)

    return F.pad(t, padding)


def concat_skip(inputs, skip, scale):
    upscaled = F.upsample_nearest(skip, scale_factor=scale)
    upscaled = centre_crop(upscaled, inputs.size())

    return torch.cat([inputs, upscaled], 1)

def selu(inputs):
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    return F.elu(inputs, alpha, inplace=True) * scale
