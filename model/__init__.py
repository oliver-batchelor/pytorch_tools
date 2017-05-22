import torch.nn.functional as F

def match_size_2d(t, sized):
    assert t.dim() == 4 and sized.dim() == 4
    dh = sized.size(2) - t.size(2)
    dw = sized.size(3) - t.size(3)

    pad = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
    return F.pad(t, pad)
