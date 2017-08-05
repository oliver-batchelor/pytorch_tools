
import torch
from torch.utils.data.sampler import Sampler

class RepeatSampler(Sampler):
    """Samples elements randomly, repeating as necessary.
    Arguments:
        num_samples (int): number of samples per epoch
        max_element (int): maximum element index

    """

    def __init__(self, num_samples, data_size):

        self.num_samples = num_samples
        self.data_size = data_size

        assert self.num_samples > 0

    def __iter__(self):
        return iter(torch.LongTensor(self.num_samples).random_(0, self.data_size))

    def __len__(self):
        return self.num_samples
