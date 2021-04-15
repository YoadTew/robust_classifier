import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
import random
import os
import sys
from datetime import datetime

from models.AffineMix import AffineMix
from models.resnet_AffineMix import resnet50
from data.data_manager import get_test_loaders

def get_args():
    parser = argparse.ArgumentParser(description="testing script",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--batch_size", "-b", type=int, default=128, help="Batch size")
    parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for dataloader")
    parser.add_argument("--data_parallel", action='store_true', help='Run on all visible gpus')

    parser.add_argument("--img_dir", default='/home/work/Datasets/ImageNet-C', help="Images dir path")

    parser.add_argument('--resume_ensemble',
                        default='experiments/Imagenet/affinemix50/randomResizeCrop/shape3uniform_resnet50_lr=0.001_train=fc/checkpoints/model_best.pth.tar',
                        type=str,
                        help='path to color model checkpoint (default: none)')

    return parser.parse_args()

def log_info(text):
    dt_string = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f'{dt_string} | {text}')
    sys.stdout.flush()

class Tester:
    def __init__(self, args, device):
        self.args = args
        self.device = device

        ensemble_model = resnet50(num_classes=1000, norm_layer=AffineMix)

        if args.resume_ensemble and os.path.isfile(args.resume_ensemble):
            print(f'Loading checkpoint {args.resume_ensemble}')

            checkpoint = torch.load(args.resume_ensemble)
            self.start_epoch = checkpoint['epoch']
            self.best_acc = checkpoint['best_prec1']
            ensemble_model.load_state_dict(checkpoint['state_dict'])

            print(f'Loaded checkpoint {args.resume_ensemble}, starting from epoch {self.start_epoch}')

        if args.data_parallel:
            ensemble_model = torch.nn.DataParallel(ensemble_model)

        self.ensemble_model = ensemble_model.to(device)
        cudnn.benchmark = True

    def do_testing(self):
        curruption_errors = []
        aug_accuracies = []
        for loader, data_name in get_test_loaders(self.args):
            accuracy = self.do_test(loader)
            aug_accuracies.append(accuracy)
            log_info(f'Dataset: {data_name}, accuracy: {accuracy}')

            if len(aug_accuracies) == 5:
                aug_name = data_name.split('_')[0]
                mean_acc = sum(aug_accuracies) / 5
                aug_accuracies = []
                curruption_errors.append(1 - mean_acc)

                log_info(f'Mean corruption: {aug_name}, accuracy: {mean_acc}, error: {1 - mean_acc}')

        print('mCE:', sum(curruption_errors) / len(curruption_errors))

    def do_test(self, loader):
        self.ensemble_model.eval()
        with torch.no_grad():
            total = len(loader.dataset)

            class_correct = 0

            for i, (inputs, targets) in enumerate(loader, 1):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                # forward
                outputs, _ = self.ensemble_model(inputs)

                _, cls_pred = outputs.max(dim=1)

                class_correct += torch.sum(cls_pred == targets)

            class_accuracy = class_correct.item() / total

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
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    main()