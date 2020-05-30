import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        loss_contrastive = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2)
            + (label)
            * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )

        return loss_contrastive


class TripletLoss(torch.nn.Module):
    """
    Triplet Loss based on minimizing distance between images
    loss = max(d(a, p) - d(a, n) + margin, 0)
    """

    def __init__(self, margin=0.2):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, a, p, n):
        d = nn.PairwiseDistance(p=2)
        distance = d(a, p) - d(a, n) + self.margin
        loss = torch.mean(torch.max(distance, torch.zeros_like(distance)))

        return loss
