
import torch
import colorsys
import random
import math

from tools import tensor
import tools.image.cv as cv
import itertools


default_colors = [
        0x00000000, 0xFFFF00FF, 0x1CE6FFFF, 0xFF34FFFF, 0xFF4A46FF, 0x008941FF, 0x006FA6FF, 0xA30059FF,
        0xFFDBE5FF, 0x7A4900FF, 0x0000A6FF, 0x63FFACFF, 0xB79762FF, 0x004D43FF, 0x8FB0FFFF, 0x997D87FF,
        0x5A0007FF, 0x809693FF, 0xFEFFE6FF, 0x1B4400FF, 0x4FC601FF, 0x3B5DFFFF, 0x4A3B53FF, 0xFF2F80FF,
        0x61615AFF, 0xBA0900FF, 0x6B7900FF, 0x00C2A0FF, 0xFFAA92FF, 0xFF90C9FF, 0xB903AAFF, 0xD16100FF,
        0xDDEFFFFF, 0x000035FF, 0x7B4F4BFF, 0xA1C299FF, 0x300018FF, 0x0AA6D8FF, 0x013349FF, 0x00846FFF,
        0x372101FF, 0xFFB500FF, 0xC2FFEDFF, 0xA079BFFF, 0xCC0744FF, 0xC0B9B2FF, 0xC2FF99FF, 0x001E09FF,
        0x00489CFF, 0x6F0062FF, 0x0CBD66FF, 0xEEC3FFFF, 0x456D75FF, 0xB77B68FF, 0x7A87A1FF, 0x788D66FF,
        0x885578FF, 0xFAD09FFF, 0xFF8A9AFF, 0xD157A0FF, 0xBEC459FF, 0x456648FF, 0x0086EDFF, 0x886F4CFF,
        0x34362DFF, 0xB4A8BDFF, 0x00A6AAFF, 0x452C2CFF, 0x636375FF, 0xA3C8C9FF, 0xFF913FFF, 0x938A81FF,
        0x575329FF, 0x00FECFFF, 0xB05B6FFF, 0x8CD0FFFF, 0x3B9700FF, 0x04F757FF, 0xC8A1A1FF, 0x1E6E00FF,
        0x7900D7FF, 0xA77500FF, 0x6367A9FF, 0xA05837FF, 0x6B002CFF, 0x772600FF, 0xD790FFFF, 0x9B9700FF,
        0x549E79FF, 0xFFF69FFF, 0x201625FF, 0x72418FFF, 0xBC23FFFF, 0x99ADC0FF, 0x3A2465FF, 0x922329FF,
        0x5B4534FF, 0xFDE8DCFF, 0x404E55FF, 0x0089A3FF, 0xCB7E98FF, 0xA4E804FF, 0x324E72FF, 0x6A3A4CFF,
        0x83AB58FF, 0x001C1EFF, 0xD1F7CEFF, 0x004B28FF, 0xC8D0F6FF, 0xA3A489FF, 0x806C66FF, 0x222800FF,
        0xBF5650FF, 0xE83000FF, 0x66796DFF, 0xDA007CFF, 0xFF1A59FF, 0x8ADBB4FF, 0x1E0200FF, 0x5B4E51FF,
        0xC895C5FF, 0x320033FF, 0xFF6832FF, 0x66E1D3FF, 0xCFCDACFF, 0xD0AC94FF, 0x7ED379FF, 0x012C58FF,
        0x7A7BFFFF, 0xD68E01FF, 0x353339FF, 0x78AFA1FF, 0xFEB2C6FF, 0x75797CFF, 0x837393FF, 0x943A4DFF,
        0xB5F4FFFF, 0xD2DCD5FF, 0x9556BDFF, 0x6A714AFF, 0x001325FF, 0x02525FFF, 0x0AA3F7FF, 0xE98176FF,
        0xDBD5DDFF, 0x5EBCD1FF, 0x3D4F44FF, 0x7E6405FF, 0x02684EFF, 0x962B75FF, 0x8D8546FF, 0x9695C5FF,
        0xE773CEFF, 0xD86A78FF, 0x3E89BEFF, 0xCA834EFF, 0x518A87FF, 0x5B113CFF, 0x55813BFF, 0xE704C4FF,
        0x00005FFF, 0xA97399FF, 0x4B8160FF, 0x59738AFF, 0xFF5DA7FF, 0xF7C9BFFF, 0x643127FF, 0x513A01FF,
        0x6B94AAFF, 0x51A058FF, 0xA45B02FF, 0x1D1702FF, 0xE20027FF, 0xE7AB63FF, 0x4C6001FF, 0x9C6966FF,
        0x64547BFF, 0x97979EFF, 0x006A66FF, 0x391406FF, 0xF4D749FF, 0x0045D2FF, 0x006C31FF, 0xDDB6D0FF,
        0x7C6571FF, 0x9FB2A4FF, 0x00D891FF, 0x15A08AFF, 0xBC65E9FF, 0xFFFFFEFF, 0xC6DC99FF, 0x203B3CFF,
        0x671190FF, 0x6B3A64FF, 0xF5E1FFFF, 0xFFA0F2FF, 0xCCAA35FF, 0x374527FF, 0x8BB400FF, 0x797868FF,
        0xC6005AFF, 0x3B000AFF, 0xC86240FF, 0x29607CFF, 0x402334FF, 0x7D5A44FF, 0xCCB87CFF, 0xB88183FF,
        0xAA5199FF, 0xB5D6C3FF, 0xA38469FF, 0x9F94F0FF, 0xA74571FF, 0xB894A6FF, 0x71BB8CFF, 0x00B433FF,
        0x789EC9FF, 0x6D80BAFF, 0x953F00FF, 0x5EFF03FF, 0xE4FFFCFF, 0x1BE177FF, 0xBCB1E5FF, 0x76912FFF,
        0x003109FF, 0x0060CDFF, 0xD20096FF, 0x895563FF, 0x29201DFF, 0x5B3213FF, 0xA76F42FF, 0x89412EFF,
        0x1A3A2AFF, 0x494B5AFF, 0xA88C85FF, 0xF4ABAAFF, 0xA3F3ABFF, 0x00C6C8FF, 0xEA8B66FF, 0x958A9FFF,
        0xBDC9D2FF, 0x9FA064FF, 0xBE4700FF, 0x658188FF, 0x83A485FF, 0x453C23FF, 0x47675DFF, 0x3A3F00FF,
        0x061203FF, 0xDFFB71FF, 0x868E7EFF, 0x98D058FF, 0x6C8F7DFF, 0xD7BFC2FF, 0x3C3E6EFF, 0xD83D66FF,
        0x2F5D9BFF, 0x6C5E46FF, 0xD25B88FF, 0x5B656CFF, 0x00B57FFF, 0x545C46FF, 0x866097FF, 0x365D25FF,
        0x252F99FF, 0x00CCFFFF, 0x674E60FF, 0xFC009CFF, 0x92896BFF]


def hex_rgba(x):
    return [(x >> 24) & 255, (x >> 16) & 255, (x >> 8) & 255, x & 255]

default_map = torch.ByteTensor(list(map(hex_rgba, default_colors)))



def combinations(total, components):
    cs = []
    for i in range(0, components):
        step = math.ceil(pow(total, 1/(components - i)))
        cs.append(step)
        total /= step

    return cs

def make_divisions(divs, total):
    if divs == 1:
        return [total]
    else:
        return [math.floor((total / (divs - 1)) * i) for i in range(0, divs)]

def take(n, iterable):
    return list(itertools.islice(iterable, n))

def make_color_map(n):
    colors = list(itertools.product(*[make_divisions(d, n) for d in combinations(n, 3)]))
    colors = colors[1:]

    random.Random(1).shuffle(colors)

    colors.insert(0, (0, 0, 0))
    rgb = torch.ByteTensor(take(n, colors))
    a   = torch.ByteTensor([0] + [255] * (n - 1))

    print(rgb.size(), a.size())

    return  torch.cat([rgb, a], 1)


#default_map = make_color_map(256)
#
#for i in range(0, 256):
#    colors = default_map[i]
#    print("{" + str(colors[0]) + ",\t" + str(colors[1]) + ",\t" + str(colors[2]) + "}, ")


def colorize(image, color_map):
    assert(image.dim() == 3 and image.size(2) == 1)

    flat_indices = image.view(-1).long()
    rgb = color_map[flat_indices]

    return rgb.view(image.size(0), image.size(1), 4)

def colorize_t(image, color_map):
    assert(image.dim() == 3 and image.size(0) == 1)
    return colorize(image.permute(1, 2, 0), color_map).permute(2, 0, 1)

def colorizer(n = 255):

    color_map = make_color_map(n)
    return lambda image: colorize(image, color_map)


def overlay_label(image, label, color_map = default_map, alpha=0.4):
    assert(image.dim() == 3 and image.size(2) == 3)

    if(label.dim() == 2):
        label = label.view(*label.size(), 1)

    assert(label.dim() == 3 and label.size(2) == 1)
    dim = (image.size(1), image.size(0))
    label = cv.resize(label, dim, interpolation = cv.inter.nearest)



    label_color = colorize(label, color_map).float()
    mask = torch.FloatTensor(image.size()).fill_(alpha)

    if(label_color.size(2) == 4):
        mask = alpha * (label_color.narrow(2, 3, 1) / 255)
        label_color = label_color.narrow(2, 0, 3)

    return (image.float() * (1 - mask) + label_color * mask).type_as(image)


def overlay_batches(images, target, cols = 6, color_map = default_map, alpha=0.4):
    images = tensor.tile_batch(images, cols)

    target = target.view(*target.size(), 1)
    target = tensor.tile_batch(target, cols)

    return overlay_label(images, target, color_map, alpha)


def counts(target, class_names = None):
    count = tensor.count_elements_sparse(target)
    if(class_names):
        return {class_names[k] if k < len(class_names) else "invalid" : n for k, n in count.items()}
    else:
        return count
