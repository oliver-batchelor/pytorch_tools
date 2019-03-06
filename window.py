
import torch


def differences(xs):
    return [t2 - t1 for t1, t2 in zip (xs, xs[1:])]

def pad(xs, n_before, n_after):
    before = xs.new(n_before).fill_(xs[0].item())
    after = xs.new(n_after).fill_(xs[-1].item())
    return  torch.cat([before, xs,  after])


def rolling_window(xs, window=5):
    n_before = window // 2

    xs = pad(xs, n_before, window - n_before - 1)
    return xs.unfold(0, window, 1)

def rolling_diff(xs, window=5):
    means = rolling_window(xs, window=window).mean(1)
    return (xs - means).abs()

def rolling_mean(xs, window=5):
    return rolling_window(xs, window=window).mean(1)

    
def masked_mean(xs, mask, window=5, clamp=True):
    xs = xs.clone().masked_fill_(~mask, 0)
    windows = rolling_window(xs, window=window)
    mask_windows = rolling_window(mask, window=window)
    sums = windows.sum(1).float()
    n = mask_windows.sum(1).float()
    if clamp:
        n.clamp_(min = 1)

    return sums / n

def masked_diff(xs, mask, window=5):
    means = masked_mean(xs, mask, window=window)
    return (xs - means).abs()


def high_variance(xs, window=5, n = 10):
    windows = rolling_window(xs, window=window)
    diffs = windows.mean(1) - xs

    return [(i.item(), v.item()) for v, i in zip(*diffs.topk(n))]

def get_clamped(xs):
    n = len(xs) - 1

    def f(i):
        return xs[max(0, min(i, n))]

    return f


def get_window(xs, i, window=5):
    
    x = []
    n_before = window // 2
    n_after = window - n_before - 1

    f = get_clamped(xs)
    return [f(i + d) for d in range(-n_before, n_after + 1)]