import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
import sys

from models.resnext import resnext29
from torchvision import datasets
from torchvision import transforms
import torchvision

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]

def get_args():
    parser = argparse.ArgumentParser(description="testing script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=1024, help="Batch size")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--data_parallel", action='store_true', help='Run on all visible gpus')

    parser.add_argument("--cifar_dir", default='/home/work/Datasets/CIFAR100', help="Images dir path")
    parser.add_argument("--corrupted_dir", default='/home/work/Datasets/CIFAR-100-C', help="Images dir path")

    parser.add_argument('--resume', default='experiments/CIFAR100/resnext29/shape=0_color=0/checkpoints/model_best.pth.tar', type=str,
                        help='path to latest checkpoint (default: none)')

    return parser.parse_args()

class Tester:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])
        self.test_data = torchvision.datasets.CIFAR100(
            self.args.cifar_dir, train=False, transform=test_transform)

        model = resnext29(num_classes=100)

        if args.resume and os.path.isfile(args.resume):
            print(f'Loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])

            print(f'Loaded checkpoint {args.resume}, starting from epoch {self.start_epoch}')

        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        self.model = model.to(device)

        cudnn.benchmark = True

    def do_testing(self):
        corruption_accs = []

        for corruption in CORRUPTIONS:
            self.test_data.data = np.load(f'{self.args.corrupted_dir}/{corruption}.npy')
            self.test_data.targets = torch.LongTensor(np.load(f'{self.args.corrupted_dir}/labels.npy'))

            test_loader = torch.utils.data.DataLoader(
                self.test_data,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.n_workers,
                pin_memory=True)

            acc = self.do_test(test_loader)
            corruption_accs.append(acc)

            print(f'Corruption: {corruption} Error is {1 - acc}')

        mean_acc = np.mean(corruption_accs)

        print(f'mCE is {1 - mean_acc}')

    def do_test(self, loader):
        self.model.eval()
        with torch.no_grad():
            total = len(loader.dataset)

            class_correct = 0

            for i, (inputs, targets) in enumerate(loader, 1):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # forward
                outputs = self.model(inputs)

                _, cls_pred = outputs.max(dim=1)

                class_correct += torch.sum(cls_pred == targets).item()

            class_accuracy = class_correct / total

            return class_accuracy

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

    tester = Tester(args, device)
    best_val_acc = tester.do_testing()

if __name__ == "__main__":
    # torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()