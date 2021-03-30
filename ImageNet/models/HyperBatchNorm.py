import torch
import torch.nn as nn
import torch.nn.functional as F

def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a, b, c * d)  # resise F_XL into \hat F_XL

    G = torch.bmm(features, features.transpose(1,2))  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class HyperBatchNorm(nn.Module):
    def __init__(self, planes):
        super(HyperBatchNorm, self).__init__()

        self.bn = nn.BatchNorm2d(planes, affine=False)

        self.weights = nn.Parameter(torch.ones(1, planes, 1, 1))
        self.biases = nn.Parameter(torch.zeros(1, planes, 1, 1))

        self.conv_weights = nn.Conv2d(planes, 1, kernel_size=1)
        self.conv_biases = nn.Conv2d(planes, 1, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        weights = self.conv_weights(x)
        weights = F.sigmoid(self.avgpool(weights))
        weights = weights * self.weights

        biases = self.conv_biases(x)
        biases = F.sigmoid(self.avgpool(biases))
        biases = biases * self.biases

        x = self.bn(x)
        x = (x * weights) + biases
        # print(x.mean().item())
        return x