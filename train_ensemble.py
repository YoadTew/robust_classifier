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
from models.ShapeNet import shapenet50, shapenet18
from models.ensemble_network import EnsembleNet
from data.data_manager import get_val_loader, get_train_loader
from data.imagenetDataset import imagenetDataset

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    parser.add_argument("--accumulate_batches", type=int, default=4, help="Number of batch to accumulate")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')

    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")

    parser.add_argument("--img_dir", default='/home/work/Datasets/Tiny-ImageNet-original', help="Images dir path")

    parser.add_argument('--use_weight_net', action='store_true', help='Load pretrain model')
    parser.add_argument('--resume_edge', default='experiments/resnet50/shape=1_color=0_loss=MSE_optim=SGD/checkpoints/model_best.pth.tar', type=str,
                        help='path to edge model checkpoint (default: none)')
    parser.add_argument('--resume_color', default='experiments/resnet50/shape=0_color=1_loss=MSE_optim=SGD/checkpoints/model_best.pth.tar', type=str,
                        help='path to color model checkpoint (default: none)')
    parser.add_argument('--resume_ensemble', default='', type=str,
                        help='path to color model checkpoint (default: none)')

    parser.add_argument("--experiment", default='experiments/ensemble50/optim=SGD',
                        help="Logs dir path")

    args = parser.parse_args()

    args.checkpoint = f'{args.experiment}/checkpoints'
    args.log_dir = f'{args.experiment}/logs'

    return args

    return parser.parse_args()

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):
    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)

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
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.start_epoch = 0
        self.best_acc = 0

        self.train_loader = get_train_loader(args, imagenetDataset)
        self.val_loader = get_val_loader(args, imagenetDataset)

        # Loads shape model
        edge_model = shapenet50(pretrained=args.pretrained, num_classes=200).to(device)
        edge_checkpoint = torch.load(args.resume_edge)
        edge_model.load_state_dict(edge_checkpoint['state_dict'])

        # Loads color model
        color_model = shapenet50(pretrained=args.pretrained, num_classes=200).to(device)
        color_checkpoint = torch.load(args.resume_color)
        color_model.load_state_dict(color_checkpoint['state_dict'])

        self.ensemble_model = EnsembleNet(edge_model, color_model, n_classes=200, use_weight_net=args.use_weight_net, device=device).to(device)

        if args.resume_ensemble and os.path.isfile(args.resume_ensemble):
            print(f'Loading checkpoint {args.resume_ensemble}')

            checkpoint = torch.load(args.resume_ensemble)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_prec1']
            self.ensemble_model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print(f'Loaded checkpoint {args.resume_ensemble}, starting from epoch {self.start_epoch}')

        # self.optimizer = torch.optim.AdamW(self.ensemble_model.get_trainable_params(), lr=args.learning_rate)
        self.optimizer = optim.SGD(self.ensemble_model.get_trainable_params(), lr=args.learning_rate, momentum=0.9)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=5)

        self.criterion = nn.CrossEntropyLoss()
        cudnn.benchmark = True
        self.writer = SummaryWriter(log_dir=str(args.log_dir))

    def _do_epoch(self, epoch_idx):
        self.ensemble_model.train()

        for batch_idx, (images, targets, _) in enumerate(self.train_loader):
            images, targets = images.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.ensemble_model(images)

            loss = self.criterion(outputs, targets)

            if batch_idx % 30 == 1:
                print(f'epoch:  {epoch_idx}/{self.args.epochs}, batch: {batch_idx}/{len(self.train_loader)}, '
                      f'loss: {loss.item()}')

            self.writer.add_scalar('loss_train', loss.item(), epoch_idx * len(self.train_loader) + batch_idx)

            loss.backward()
            self.optimizer.step()

        self.ensemble_model.eval()

        with torch.no_grad():
            total = len(self.val_loader.dataset)
            class_correct = self.do_test(self.val_loader)
            class_acc = float(class_correct) / total
            print(f'Validation Accuracy: {class_acc}')

            is_best = False
            if class_acc > self.best_acc:
                self.best_acc = class_acc
                is_best = True

            checkpoint_name = f'checkpoint_{epoch_idx + 1}_acc_{round(class_acc, 3)}.pth.tar'
            print(f'Saving {checkpoint_name} to dir {self.args.checkpoint}')
            save_checkpoint({
                'epoch': epoch_idx + 1,
                'state_dict': self.ensemble_model.state_dict(),
                'best_prec1': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, checkpoint=self.args.checkpoint, filename=checkpoint_name)

            self.writer.add_scalar('val_accuracy', class_acc, epoch_idx)

    def do_test(self, loader):
        class_correct = 0

        for i, (inputs, targets, _) in enumerate(loader, 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # forward
            outputs = self.ensemble_model(inputs)

            _, cls_pred = outputs.max(dim=1)

            class_correct += torch.sum(cls_pred == targets)

        return class_correct

    def do_training(self):
        for self.current_epoch in range(self.start_epoch, self.args.epochs):
            self._do_epoch(self.current_epoch)
            self.scheduler.step()

        self.writer.close()

        return self.best_acc

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