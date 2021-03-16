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
from data.shape.data_manager import get_val_loader, get_train_loader

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    # parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')

    parser.add_argument("--learning_rate", "-l", type=float, default=.001, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--shape_loss_weight", type=float, default=1., help="Shape loss weight")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")

    parser.add_argument("--img_dir", default='/home/work/Datasets/Tiny-ImageNet-original', help="Images dir path")

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--checkpoint", default='checkpoints', help="Logs dir path")
    parser.add_argument("--log_dir", default='logs', help="Logs dir path")
    parser.add_argument("--log_prefix", default='', help="Logs dir path")

    return parser.parse_args()

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
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.start_epoch = 0
        self.best_acc = 0

        model = resnet18(pretrained=args.pretrained, num_classes=200)
        self.model = model.to(device)

        self.train_loader = get_train_loader(args)
        self.val_loader = get_val_loader(args)

        # self.optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=int(args.epochs * .3))
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.args.epochs)

        self.criterion = nn.CrossEntropyLoss()
        self.shape_criterion = nn.MSELoss()

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

        for batch_idx, (images, targets, sobels) in enumerate(self.train_loader):
            images, targets, sobels = images.to(self.device), targets.to(self.device), sobels.to(self.device)

            self.optimizer.zero_grad()
            outputs, imgs_logits = self.model(images)
            sobel_outputs, sobels_logits = self.model(sobels)
            # outputs = torch.squeeze(outputs)

            cls_loss = self.criterion(outputs, targets)
            shape_loss = self.shape_criterion(imgs_logits, sobels_logits)

            loss = cls_loss + self.args.shape_loss_weight * shape_loss

            if batch_idx % 30 == 1:
                print(f'epoch:  {epoch_idx}/{self.args.epochs}, batch: {batch_idx}/{len(self.train_loader)}, '
                      f'loss: {loss.item()}, cls_loss: {cls_loss.item()}, shape_loss: {shape_loss.item()}')

            self.writer.add_scalar('loss_train', loss.item(), epoch_idx * len(self.train_loader) + batch_idx)
            self.writer.add_scalar('cls_loss_train', cls_loss.item(), epoch_idx * len(self.train_loader) + batch_idx)
            self.writer.add_scalar('shape_loss_train', shape_loss.item(), epoch_idx * len(self.train_loader) + batch_idx)

            loss.backward()
            self.optimizer.step()

        self.model.eval()
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
                'state_dict': self.model.state_dict(),
                'best_prec1': self.best_acc,
                'optimizer': self.optimizer.state_dict(),
            }, is_best, checkpoint=self.args.checkpoint, filename=checkpoint_name)

            self.writer.add_scalar('val_accuracy', class_acc, epoch_idx)

    def do_test(self, loader):
        class_correct = 0

        for i, (inputs, targets, _) in enumerate(loader, 1):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            # forward
            outputs, _ = self.model(inputs)

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