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

from models.resnet import resnet18
from models.ensemble_network import EnsembleNet

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    # parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')

    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")

    parser.add_argument("--img_dir", default='/home/work/Datasets/Tiny-ImageNet-original', help="Images dir path")

    parser.add_argument('--resume_edge', default='', type=str,
                        help='path to edge model checkpoint (default: none)')
    parser.add_argument('--resume_color', default='', type=str,
                        help='path to color model checkpoint (default: none)')
    parser.add_argument('--resume_ensemble', default='', type=str,
                        help='path to color model checkpoint (default: none)')

    parser.add_argument("--checkpoint", default='checkpoints', help="Logs dir path")
    parser.add_argument("--log_dir", default='logs/ensemble', help="Logs dir path")
    parser.add_argument("--log_prefix", default='', help="Logs dir path")

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'model_best.pth.tar'))

def save_args_json(args):
    args_dict = vars(args)

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    with open(f'{args.log_dir}/args.json', 'w') as outfile:
        json.dump(args_dict, outfile, indent=4, sort_keys=True)

class Trainer:
    def __init__(self):
        pass

def main():
    args = get_args()

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if sys.gettrace() is None:  # Not debug
        args.n_workers = 4
    else:  # Debug
        args.n_workers = 0

    save_args_json(args)
    trainer = Trainer(args, device)
    best_val_acc = trainer.do_training()

if __name__ == "__main__":
    # torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()