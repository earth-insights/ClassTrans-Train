import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def iou(pr, gt, eps=1e-7, threshold=None):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    intersection = torch.sum((gt * pr).float())
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps
    return (intersection + eps) / union

def mIoU(pr, gt, eps=1e-7, n_classes=9):
    pr = F.softmax(pr, dim=1)              
    pr = torch.argmax(pr, dim=1).squeeze(1)
    if len(gt.shape) == 4:
        gt = torch.argmax(gt, dim=1).squeeze(1)
    iou_per_class = []

    pr = pr.contiguous().view(-1)
    gt = gt.contiguous().view(-1)

    for sem_class in range(1, n_classes): # background is not included
        pr_inds = (pr == sem_class)
        gt_inds = (gt == sem_class)
        if gt_inds.long().sum().item() == 0:
            iou_per_class.append(np.nan)
        else: 
            intersect = torch.logical_and(pr_inds, gt_inds).sum().float().item()
            union = torch.logical_or(pr_inds, gt_inds).sum().float().item()
            iou = (intersect + eps) / (union + eps)
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class)

def fscore(pr, gt, beta=1, eps=1e-7, threshold=None):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp
    score = ((1 + beta ** 2) * tp + eps) / (
        (1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps
    )
    return score


class Fscore(nn.Module):
    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "Fscore"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.softmax(input, dim=1).argmax(dim=1)
        scores = []
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :].sigmoid()
            ygt = target[:, i, :, :]
            scores.append(fscore(ypr, ygt, threshold=self.threshold))
        return sum(scores) / len(scores)


class IoU(nn.Module):
    def __init__(self, class_weights=1.0, threshold=None):
        super().__init__()
        self.class_weights = torch.tensor(class_weights)
        self.threshold = threshold
        self.name = "IoU"

    @torch.no_grad()
    def forward(self, input, target):
        input = torch.softmax(input, dim=1)
        scores = []
        for i in range(1, input.shape[1]):  # background is not included
            ypr = input[:, i, :, :].sigmoid() > 0.5
            ygt = target[:, i, :, :]
            scores.append(iou(ypr, ygt, threshold=self.threshold))
        return sum(scores) / len(scores)

class IoU2(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "mIoU"

    @torch.no_grad()
    def forward(self, input, target):
        score = mIoU(input, target, n_classes=input.shape[1])
        return torch.tensor(score)


def mIoU3(pr, gt, eps=1e-7, n_classes=9):
    iou_per_class = []

    if not isinstance(pr, torch.Tensor):
        pr = torch.from_numpy(pr)
    if not isinstance(gt, torch.Tensor):
        gt = torch.from_numpy(gt)

    pr = pr.contiguous().view(-1)
    gt = gt.contiguous().view(-1)

    for sem_class in range(1, n_classes): # background is not included
        pr_inds = (pr == sem_class)
        gt_inds = (gt == sem_class)
        if gt_inds.long().sum().item() == 0:
            iou_per_class.append(np.nan)
        else: 
            intersect = torch.logical_and(pr_inds, gt_inds).sum().float().item()
            union = torch.logical_or(pr_inds, gt_inds).sum().float().item()
            iou = (intersect + eps) / (union + eps)
            iou_per_class.append(iou)
    return np.nanmean(iou_per_class)

class IoU3(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "mIoU"

    @torch.no_grad()
    def forward(self, input, target):
        score = mIoU3(input, target, n_classes=input.shape[1])
        return torch.tensor(score)