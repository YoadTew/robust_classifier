import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperBatchNorm(nn.Module):
    def __init__(self, planes):
        super(HyperBatchNorm, self).__init__()

        self.bn = nn.BatchNorm2d(planes, affine=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_weights = nn.Linear(planes, planes)
        self.fc_biases = nn.Linear(planes, planes)

    def forward(self, x):
        pooled = self.avgpool(x)
        pooled = pooled.flatten(1)

        weights = self.fc_weights(pooled)
        biases = self.fc_biases(pooled)

        x = self.bn(x)

        x = (x * weights) + biases

        return x