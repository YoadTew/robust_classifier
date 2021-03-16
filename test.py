import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
import sys

from models.resnet import resnet18
from data.data_manager import get_test_loaders

def get_args():
    parser = argparse.ArgumentParser(description="testing script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=32, help="Batch size")
    # parser.add_argument("--image_size", type=int, default=222, help="Image size")
    parser.add_argument('--pretrained', action='store_true', help='Load pretrain model')
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")

    parser.add_argument("--img_dir", default='/home/work/Datasets/ImageNet-C/weather', help="Images dir path")

    parser.add_argument('--resume', default='', type=str,
                        help='path to latest checkpoint (default: none)')
    parser.add_argument("--checkpoint", default='checkpoints', help="Logs dir path")
    parser.add_argument("--log_dir", default='logs', help="Logs dir path")
    parser.add_argument("--log_prefix", default='', help="Logs dir path")

    return parser.parse_args()

class Tester:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        model = resnet18(pretrained=args.pretrained, num_classes=200)
        self.model = model.to(device)

        if args.resume and os.path.isfile(args.resume):
            print(f'Loading checkpoint {args.resume}')

            checkpoint = torch.load(args.resume)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_prec1']
            self.model.load_state_dict(checkpoint['state_dict'])

            print(f'Loaded checkpoint {args.resume}, starting from epoch {self.start_epoch}')

        cudnn.benchmark = True

    def do_testing(self):
        for loader, data_name in get_test_loaders(self.args):
            accuracy = self.do_test(loader)
            print(f'Dataset: {data_name}, accuracy: {accuracy}')

        # accuracy = do_test(self.test_loader)

    def do_test(self, loader):
        self.model.eval()
        with torch.no_grad():
            total = len(loader.dataset)

            class_correct = 0

            for i, (inputs, targets) in enumerate(loader, 1):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # forward
                outputs, _ = self.model(inputs)

                _, cls_pred = outputs.max(dim=1)

                class_correct += torch.sum(cls_pred == targets)

            class_accuracy = class_correct / total

            return class_accuracy

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

    tester = Tester(args, device)
    best_val_acc = tester.do_testing()

if __name__ == "__main__":
    # torchfunc.cuda.reset()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()