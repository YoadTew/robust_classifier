import torch
import torch.nn as nn
import torch.nn.functional as F

class EnsembleBatchNorm(nn.Module):
    def __init__(self, planes):
        super(EnsembleBatchNorm, self).__init__()

        self.bn_shape = nn.BatchNorm2d(planes)
        self.bn_color = nn.BatchNorm2d(planes)

        self.convex_weights = nn.Parameter(torch.ones(1, 2))

    def forward(self, x):

        shape = x.size()

        x_1 = self.bn_shape(x)
        x_2 = self.bn_color(x)

        x = torch.matmul(self.convex_weights, torch.stack([x_1.view(-1), x_2.view(-1)]))
        x = x.view(shape)

        return x