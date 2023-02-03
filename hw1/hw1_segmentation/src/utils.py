import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mean_iou_score(preds: np, labels: np) -> float:
    """
    Compute mean IoU score over 6 classes
    """

    mean_iou = 0
    for i in range(6):
        tp_fp = np.sum(preds == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((preds == i) * (labels == i))
        # print("numbers of tp, tp_fp, tp_fn", tp, tp_fp, tp_fn)
        iou = tp / (tp_fp + tp_fn - tp + 1e-6)
        mean_iou += iou / 6
    #     print("class #%d : %1.5f" % (i, iou))
    # print("\nmean_iou: ", mean_iou, "\n")

    return mean_iou


class SoftIoULoss(nn.Module):
    def __init__(self, n_classes: int, device: torch.device):
        super(SoftIoULoss, self).__init__()
        self.n_classes = n_classes
        self.device = device

    @staticmethod
    def to_one_hot(tensor, n_classes):
        n, h, w = tensor.size()
        tensor = tensor.to("cpu")
        one_hot = torch.zeros(n, n_classes, h, w).scatter_(
            1, tensor.view(n, 1, h, w).to(torch.int64), 1
        )
        return one_hot

    def forward(self, input, target):
        # logit => N x Classes x H x W
        # target => N x H x W

        N = len(input)

        pred = F.softmax(input, dim=1)
        target_onehot = self.to_one_hot(target, self.n_classes).to(self.device)

        # Numerator Product
        inter = pred * target_onehot
        # Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.n_classes, -1).sum(2)

        # Denominator
        union = pred + target_onehot - (pred * target_onehot)
        # Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.n_classes, -1).sum(2)

        loss = inter / (union + 1e-16)

        # Return average loss over classes and batch
        return -loss.mean()
