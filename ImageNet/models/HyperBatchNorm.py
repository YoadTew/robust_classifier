import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperBatchNorm(nn.Module):
    def __init__(self, planes, embedding_size):
        super(HyperBatchNorm, self).__init__()

        self.bn = nn.BatchNorm2d(planes, affine=False)
        self.fc_weights = nn.Linear(embedding_size, planes)
        self.fc_biases = nn.Linear(embedding_size, planes)

    def forward(self, x, embedding):

        x = self.bn(x)

        weights = self.fc_weights(embedding)
        biases = self.fc_biases(embedding)

        x = (x * weights) + biases

        return x