import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import numpy as np

from models.resnet import resnet18

class WeightNet(nn.Module):
    def __init__(self, n_weights):
        super(WeightNet, self).__init__()

        self.resnet = resnet18(pretrained=False, num_classes=n_weights)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x, _ = self.resnet(x)
        x = self.sigmoid(x)
        # x = F.softmax(x, dim=-1)

        return x

class EnsembleNet(nn.Module):

    def __init__(self, edge_model, color_model, use_weight_net=True, n_classes=200, device='cuda'):
        super(EnsembleNet, self).__init__()

        self.n_classifiers = 2
        self.use_weight_net = use_weight_net

        if use_weight_net:
            self.weights_net = WeightNet(n_weights=self.n_classifiers)
        else:
            self.weights = torch.ones(self.n_classifiers, device=device)

        self.edge_model = edge_model
        self.color_model = color_model
        self.dropout = nn.Dropout(p=0.25)

        self.project = nn.Linear(self.edge_model.fc.in_features * self.n_classifiers, n_classes)

    def forward(self, x):
        _, features_edge = self.edge_model(x)
        _, features_color = self.color_model(x)

        features_edge = features_edge['representation']
        features_color = features_color['representation']

        if self.use_weight_net:
            weights = self.weights_net(x)
            weights = weights.unsqueeze(-1)
        else:
            weights = self.weights
            # weights = self.dropout(weights)
            weights = weights.view(1, -1, 1)

        weighted_features = weights * torch.stack([features_edge, features_color], dim=1)
        weighted_features = weighted_features.view(features_edge.size(0), -1)

        logits = self.project(weighted_features)

        return logits

    def get_trainable_params(self):
        for param in self.edge_model.parameters():
            param.requires_grad = False
        for param in self.color_model.parameters():
            param.requires_grad = False

        return filter(lambda p: p.requires_grad, self.parameters())
