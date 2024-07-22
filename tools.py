import torch
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms

def kl_div(logit1, logit2):
    return F.kl_div(F.log_softmax(logit1, dim=1), F.softmax(logit2, dim=1), reduction='batchmean')

def mixup_data(x, y, alpha=1.0):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def label_smoothing(onehot, n_classes, factor):
    return onehot * factor + (onehot - 1) * ((factor - 1)/(n_classes - 1))


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])


print(torch.__version__)