import torch
from tools.image import index_map, cv


def split(t, dim = 0):
    return [chunk.squeeze(dim) for chunk in t.chunk(t.size(dim), dim)]




def tile_batch(t, cols=int(6)):
    assert(t.dim() == 4)

    if(t.size(0)) == 1:
        return t[0]

    h = t.size(1)
    w = t.size(2)

    rows = (t.size(0) - 1) // cols + 1
    out = t.new(rows * h, cols * w, t.size(3)).fill_(0)

    for i in range(0, t.size(0)):
        x = i % cols
        y = i // cols

        tile = out.narrow(1, x * w, w).narrow(0, y * h, h)
        tile.copy_(t[i])

    return out


def count_elements(indices, num_entries  = None):
    indices = indices.long().view(-1)

    num_entries = num_entries or (indices.max() + 1)
    c = torch.LongTensor(num_entries).fill_(0)

    ones = torch.LongTensor([1]).expand(indices.size(0))
    return c.index_add_(0, indices, ones)


def insert_size(s, dim, n):
    size = list(s)
    size.insert(dim, n)

    return torch.Size(size)


def one_hot(labels, classes, dim = 1):

    expanded = labels.view(insert_size(labels.size(), dim, 1))
    target = labels.new(insert_size(labels.size(), dim, classes))

    return target.zero_().scatter_(dim, expanded, 1)

def count_elements_sparse(indices, num_entries  = None):
    elems = count_elements(indices, num_entries)
    inds = torch.nonzero(elems).squeeze(1)

    d = {}

    for i in inds:
        d[i] = elems[i]
    return d

def index(table, inds):
    flat_indices = inds.view(-1).long()
    flat_result = table.index(flat_indices)
    return flat_result.view(inds.size())

def show_batch_t(data, cols=int(6), scale=1):
    return show_batch(data.permute(0, 2, 3, 1), cols=cols, scale=scale)


def show_batch(t, cols=int(6), scale=1):
    tiled = tile_batch(t, cols)
    tiled = cv.resize (tiled, (tiled.size(0) * scale, tiled.size(1) * scale), interpolation = cv.INTER_NEAREST)

    return cv.display(tiled)

def show_indexed_batch(t, cols=int(6)):
    colorizer = index_map.colorizer_t(255)
    tiled = tile_batch(t, cols)

    color = colorizer(tiled)
    return cv.display(color)


def centre_crop(t, size):
    d = t.dim()

    dw = t.size(d-1) - size[3]
    dh = t.size(d-2) - size[2]


    if not (dw >= 0 and dh >= 0):
        print(t.size(), size)

    return t.narrow(d-1, dw//2, size[3]).narrow(d-2, dh//2, size[2]).contiguous()


def pluck(t, indices, dim1=0, dim2=1):
    n = indices.size(0)
    #print(indices.dim(), indices.size(), t.size(), dim1, dim2)
    assert indices.dim() == 1 and t.size(dim1) == n
    t = t.contiguous()
    indices = indices.long()
    r = torch.arange(0, n).long().mul_(t.stride(dim1))
    indices = indices.mul(t.stride(dim2)) + r
    return t.view(-1).index(indices)
