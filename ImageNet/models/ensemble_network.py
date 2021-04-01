import torch
import torch.nn as nn

from ImageNet.models.resnet import resnet18

class EnsembleNet(nn.Module):

    def __init__(self, edge_model, color_model, use_weight_net=True, n_classes=200, device='cuda'):
        super(EnsembleNet, self).__init__()

        self.n_classifiers = 2
        self.use_weight_net = use_weight_net

        self.edge_model = edge_model
        self.color_model = color_model
        self.dropout = nn.Dropout(p=0.25)

        self.project = nn.Linear(self.edge_model.fc.in_features * self.n_classifiers, n_classes)
    def forward(self, x):
        _, features_edge = self.edge_model(x)
        _, features_color = self.color_model(x)

        features_edge = features_edge['representation']
        features_color = features_color['representation']

        weights = torch.ones((x.size(0), self.n_classifiers, 1), device=x.device)
        weights = self.dropout(weights)

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
