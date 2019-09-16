from torch.utils.data.sampler import Sampler
import torch.utils.data as data

import torch

class RandomSampler(Sampler):
    """Samples elements randomly from a list, rather than providing indicies.
    Arguments:
        items (list): items to randomly sample
        num_samples (int): number of samples to iterate

    """

    def __init__(self, items, num_samples):

        self.items = items
        self.num_samples = num_samples

        assert self.num_samples > 0

    def __iter__(self):
        samples = torch.LongTensor(self.num_samples).random_(0, len(self.items)) if len(self.items) > 0 else torch.LongTensor()
        return (self.items[i] for i in samples)


    def __len__(self):
        return self.num_samples

class ListSampler(Sampler):
    """Samples elements directly from a list.
    Arguments:
        items (list): items to iterate
    """

    def __init__(self, items, num_samples):
        self.items = items


    def __iter__(self):
        return iter(self.items)


    def __len__(self):
        return len(self.items)


class Loader(data.Dataset):

    def __init__(self, loader, transform=None):

        self.transform = transform
        self.loader = loader

    def __getitem__(self, item):

        image = self.loader(item)
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return 0
