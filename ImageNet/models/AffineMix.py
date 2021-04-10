import torch
import torch.nn as nn
import torch.nn.functional as F

class AffineMix(nn.Module):
    def __init__(self, planes):
        super(AffineMix, self).__init__()

        self.bn_shape = nn.BatchNorm2d(planes)
        self.bn_color = nn.BatchNorm2d(planes)

        self.convex_weights = nn.Parameter(torch.ones(2) * 0.5)

    def forward(self, x):

        x_1 = self.bn_shape(x)
        x_2 = self.bn_color(x)

        x = self.convex_weights[0] * x_1 + self.convex_weights[1] * x_2

        return x