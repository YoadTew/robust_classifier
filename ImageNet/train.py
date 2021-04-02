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
from models.HyperBatchNorm import HyperBatchNorm

def get_args():
    parser = argparse.ArgumentParser(description="training script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')

    parser.add_argument("--batch_size", "-b", type=int, default=256, help="Batch size")
    parser.add_argument("--img_size", type=int, default=256, help="Image size to resize before cropping 224")
    parser.add_argument("--MILESTONES", nargs='*', type=int, default=[25, 50, 75], help="Learning rate")
    parser.add_argument("--learning_rate", "-l", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of epochs")
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,  help='weight decay')

    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--data_parallel", action='store_true', help='Run on all visible gpus')

    parser.add_argument("--shape_loss_weight", type=float, default=0., help="Shape loss weight")
    parser.add_argument("--color_loss_weight", type=float, default=0., help="Color loss weight")
    parser.add_argument("--distance_criterion", type=str, default='MSE', help="MSE or cosine")
    parser.add_argument("--norm_layer", type=str, choices=[None, 'HyperBatchNorm'], help="Use a different norm layer")
    parser.add_argument("--train_only_bn", action='store_true', help="Train only batchNorm layers")


    parser.add_argument("--img_dir", default='/home/work/Datasets/Tiny-ImageNet-original', help="Images dir path")

    parser.add_argument("--save_checkpoint_interval", type=int, default=10, help="Save checkpoints every i epochs")
    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--experiment", default='experiments/resnet50/dummy',
                        help="Logs dir path")

    args = parser.parse_args()

    args.checkpoint = f'{args.experiment}/checkpoints'
    args.log_dir = f'{args.experiment}/logs'

    if args.norm_layer == 'HyperBatchNorm':
        args.norm_layer = HyperBatchNorm

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

def cosine_loss(criterion, pred, target, device='cuda'):
    indication = torch.tensor(1, device=device)

    return criterion(pred, target, indication)

class Trainer:
    def __init__(self, args, device):
        self.args = args
        self.device = device
        self.start_epoch = 0
        self.best_acc = 0

        self.use_shape = (self.args.shape_loss_weight > 0)
        self.use_color = (self.args.color_loss_weight > 0)

        self.train_loader = get_train_loader(args, imagenetDataset, use_sobel=self.use_shape, use_color=self.use_color)
        self.val_loader = get_val_loader(args, imagenetDataset)

        model = resnet50(pretrained=args.pretrained, num_classes=200, norm_layer=args.norm_layer)
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'Parameter count: {param_count:,}')

        if args.train_only_bn:
            trainable_params = model.get_bn_params_freeze_rest()
        else:
            trainable_params = model.parameters()

        self.optimizer = optim.SGD(trainable_params, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=args.MILESTONES, gamma=0.1)

        if args.resume and os.path.isfile(args.resume):
            print(f'Loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            print(f'Loaded checkpoint {args.resume}, starting from epoch {self.start_epoch}')

        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        self.model = model.to(device)

        self.criterion = nn.CrossEntropyLoss()

        if args.distance_criterion == 'MSE':
            self.shape_criterion = nn.MSELoss()
            self.distance_loss_func = MSE_loss
        elif args.distance_criterion == 'cosine':
            self.shape_criterion = nn.CosineEmbeddingLoss()
            self.distance_loss_func = cosine_loss

        cudnn.benchmark = True
        self.writer = SummaryWriter(log_dir=str(args.log_dir))

    def _do_epoch(self, epoch_idx):
        self.model.train()

        for batch_idx, (images, targets, extra_data) in enumerate(self.train_loader):
            self.optimizer.zero_grad()
            images, targets = images.to(self.device), targets.to(self.device)

            if self.use_shape:
                sobels = extra_data['sobel'].to(self.device)
            if self.use_color:
                colored = extra_data['color'].to(self.device)

            outputs, img_activations = self.model(images)
            img_features = img_activations['representation']

            loss = 0.

            cls_loss = self.criterion(outputs, targets)
            loss += cls_loss

            if self.use_shape:
                with torch.no_grad():
                    _, img_activations = self.model(sobels)
                    sobel_ftrs = img_activations['representation'].detach()

                shape_loss = self.distance_loss_func(self.shape_criterion, img_features, sobel_ftrs, self.device)
                self.writer.add_scalar('shape_loss_train', shape_loss.item(),
                                       epoch_idx * len(self.train_loader) + batch_idx)
                loss += self.args.shape_loss_weight * shape_loss

            if self.use_color:
                with torch.no_grad():
                    _, img_activations = self.model(colored)
                    color_ftrs = img_activations['representation'].detach()

                color_loss = self.distance_loss_func(self.shape_criterion, img_features, color_ftrs, self.device)
                self.writer.add_scalar('color_loss_train', color_loss.item(),
                                       epoch_idx * len(self.train_loader) + batch_idx)

                loss += self.args.color_loss_weight * color_loss

            if batch_idx % 100 == 1:
                print(f'epoch:  {epoch_idx}/{self.args.epochs}, batch: {batch_idx}/{len(self.train_loader)}, '
                      f'loss: {loss.item()}, cls_loss: {cls_loss.item()}, extra_losses: {loss.item() - cls_loss.item()}')

            n_iter = epoch_idx * len(self.train_loader) + batch_idx
            self.writer.add_scalar('loss_train', loss.item(), n_iter)
            self.writer.add_scalar('cls_loss_train', cls_loss.item(), n_iter)

            loss.backward()
            self.optimizer.step()


            if not self.args.data_parallel:
                last_layer = list(self.model.children())[-1]
            else:
                last_layer = list(self.model.module.children())[-1]

            for name, para in last_layer.named_parameters():
                if 'weight' in name:
                    self.writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
                if 'bias' in name:
                    self.writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

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

    if sys.gettrace() is not None:  # debug
        print('Debug mode!')
        args.n_workers = 0

    save_args_json(args)
    trainer = Trainer(args, device)
    best_val_acc = trainer.do_training()

if __name__ == "__main__":
    # torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()