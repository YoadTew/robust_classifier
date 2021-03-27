import argparse

import torch
from torch import nn, optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import random
import os
import shutil
import sys
import json

from data.augmix_data_manager import get_val_loader, get_train_loader
from data.CIFAR100Dataset import CIFAR100Dataset
from models.resnext import resnext29


def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')

    parser.add_argument("--learning_rate", "-l", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=200, help="Number of epochs")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--data_parallel", action='store_true', help='Run on all visible gpus')

    parser.add_argument("--shape_loss_weight", type=float, default=0., help="Shape loss weight")
    parser.add_argument("--color_loss_weight", type=float, default=0., help="Color loss weight")
    parser.add_argument("--distance_criterion", type=str, default='MSE', help="MSE or cosine")

    # AugMix options
    parser.add_argument(
        '--mixture-width',
        default=3,
        type=int,
        help='Number of augmentation chains to mix per augmented example')
    parser.add_argument(
        '--mixture-depth',
        default=-1,
        type=int,
        help='Depth of augmentation chains. -1 denotes stochastic depth in [1, 3]')
    parser.add_argument(
        '--aug-severity',
        default=3,
        type=int,
        help='Severity of base augmentation operators')
    parser.add_argument(
        '--no-jsd',
        '-nj',
        action='store_true',
        help='Turn off JSD consistency loss.')
    parser.add_argument(
        '--all-ops',
        '-all',
        action='store_true',
        help='Turn on all operations (+brightness,contrast,color,sharpness).')

    parser.add_argument("--img_dir", default='/home/work/Datasets/CIFAR100/cifar-100', help="Images dir path")

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--experiment", default='experiments/CIFAR100/augmix_resnext29/shape=0_color=0',
                        help="Logs dir path")
    parser.add_argument("--save_checkpoint_interval", type=int, default=10, help="Save checkpoints every i epochs")

    args = parser.parse_args()

    args.checkpoint = f'{args.experiment}/checkpoints'
    args.log_dir = f'{args.experiment}/logs'

    return args

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

def MSE_loss(criterion, pred, target, device='cuda'):
    return criterion(pred, target)

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 +
                                             np.cos(step / total_steps * np.pi))

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.start_epoch = 0
        self.best_acc = 0

        self.use_shape = (self.args.shape_loss_weight > 0)
        self.use_color = (self.args.color_loss_weight > 0)

        model = resnext29(num_classes=100)

        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        self.model = model.to(device)

        self.train_loader = get_train_loader(args, CIFAR100Dataset, use_sobel=self.use_shape, use_color=self.use_color)
        self.val_loader = get_val_loader(args, CIFAR100Dataset)

        self.optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4, nesterov=True)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                args.epochs * len(self.train_loader),
                1,  # lr_lambda computes multiplicative factor
                1e-6 / args.learning_rate))
        self.criterion = nn.CrossEntropyLoss()

        self.shape_criterion = nn.MSELoss()
        self.distance_loss_func = MSE_loss

        if args.resume and os.path.isfile(args.resume):
            print(f'Loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print(f'Loaded checkpoint {args.resume}, starting from epoch {self.start_epoch}')

        cudnn.benchmark = True
        self.writer = SummaryWriter(log_dir=str(args.log_dir))

    def _do_epoch(self, epoch_idx):
        self.model.train()

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            self.optimizer.zero_grad()

            if self.args.no_jsd:
                images, targets = images.to(self.device), targets.to(self.device)
                logits = self.model(images)
                loss = F.cross_entropy(logits, targets)
            else:
                images_all = torch.cat(images, 0).to(self.device)
                targets = targets.to(self.device)
                logits_all = self.model(images_all)
                logits_clean, logits_aug1, logits_aug2 = torch.split(
                    logits_all, images[0].size(0))

                # Cross-entropy is only computed on clean images
                loss = F.cross_entropy(logits_clean, targets)

                p_clean, p_aug1, p_aug2 = F.softmax(
                    logits_clean, dim=1), F.softmax(
                    logits_aug1, dim=1), F.softmax(
                    logits_aug2, dim=1)

                # Clamp mixture distribution to avoid exploding KL divergence
                p_mixture = torch.clamp((p_clean + p_aug1 + p_aug2) / 3., 1e-7, 1).log()
                loss += 12 * (F.kl_div(p_mixture, p_clean, reduction='batchmean') +
                              F.kl_div(p_mixture, p_aug1, reduction='batchmean') +
                              F.kl_div(p_mixture, p_aug2, reduction='batchmean')) / 3.


            if batch_idx % 100 == 1:
                print(f'epoch:  {epoch_idx}/{self.args.epochs}, batch: {batch_idx}/{len(self.train_loader)}, '
                      f'loss: {loss.item()}')

            n_iter = epoch_idx * len(self.train_loader) + batch_idx
            self.writer.add_scalar('loss_train', loss.item(), n_iter)

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            last_layer = list(self.model.children())[-1]
            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    self.writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    self.writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        with torch.no_grad():
            total = len(self.val_loader.dataset)
            class_correct = self.do_test(self.val_loader)
            class_acc = float(class_correct) / total
            print(f'Validation Accuracy: {class_acc}')

            is_best = False
            if class_acc > self.best_acc:
                self.best_acc = class_acc
                is_best = True

            if is_best or (epoch_idx + 1) % self.args.save_checkpoint_interval == 0:
                checkpoint_name = f'checkpoint_{epoch_idx + 1}_acc_{round(class_acc, 3)}.pth.tar'
                print(f'Saving {checkpoint_name} to dir {self.args.checkpoint}')

                if self.args.data_parallel:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                save_checkpoint({
                    'epoch': epoch_idx + 1,
                    'state_dict': state_dict,
                    'best_prec1': self.best_acc,
                    'optimizer': self.optimizer.state_dict(),
                }, is_best, checkpoint=self.args.checkpoint, filename=checkpoint_name)

            self.writer.add_scalar('val_accuracy', class_acc, epoch_idx)

    def do_test(self, loader):
        class_correct = 0

        for i, (inputs, targets, _) in enumerate(loader, 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # forward
            outputs = self.model(inputs)

            _, cls_pred = outputs.max(dim=1)

            class_correct += torch.sum(cls_pred == targets)

        return class_correct

    def do_training(self):
        for self.current_epoch in range(self.start_epoch, self.args.epochs):
            self._do_epoch(self.current_epoch)

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
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()