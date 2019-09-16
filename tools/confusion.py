import torch
import tools.tensor as tensor

def confusion_matrix(pred, target, num_classes):
    assert pred.size() == target.size(), "prediction must match target size"
    pred = pred.view(-1)
    target = target.view(-1)

    mask = (target < num_classes).long()
    n = num_classes * num_classes

    return tensor.count_elements(pred + (target * mask * num_classes) + mask, n + 1).narrow(0, 1, n).view(num_classes, num_classes)


def confusion_zero(n):
    return torch.LongTensor (n, n).fill_(0)
