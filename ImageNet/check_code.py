import argparse

import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import random
import os
import shutil
import sys
import json

from models.resnet import resnet50
from data.data_manager import get_val_loader, get_train_loader
from data.imagenetDataset import imagenetDataset
from models.EnsembleBatchNorm import EnsembleBatchNorm

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--resume_edge', default='experiments/ImageNetSubset/resnet50/shape=1_color=0_pretrained_lr=0.005_trainBN/checkpoints/model_best.pth.tar', type=str,
                        help='path to edge model checkpoint (default: none)')
    parser.add_argument('--resume_color', default='experiments/ImageNetSubset/resnet50/shape=0_color=1_pretrained_lr=0.005_trainBN/checkpoints/model_best.pth.tar', type=str,
                        help='path to color model checkpoint (default: none)')

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    edge_model = resnet50(num_classes=200)
    edge_checkpoint = torch.load(args.resume_edge)
    edge_state_dict = edge_checkpoint['state_dict']
    edge_model.load_state_dict(edge_state_dict)

    color_model = resnet50(num_classes=200)
    color_checkpoint = torch.load(args.resume_color)
    color_state_dict = color_checkpoint['state_dict']
    color_model.load_state_dict(color_state_dict)

    ensemble_model = resnet50(num_classes=200, norm_layer=EnsembleBatchNorm)
    ensemble_model.load_batchEnsemble_state_dict(edge_model, color_model)

    print('yay')

if __name__ == "__main__":
    # torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()