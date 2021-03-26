from torch.utils import data
import torchvision.transforms as transforms
import os

import glob

CIFAR100_TRAIN_MEAN = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
CIFAR100_TRAIN_STD = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]

CIFAR100_TEST_MEAN = [0.5088964127604166, 0.48739301317401956, 0.44194221124387256]
CIFAR100_TEST_STD = [0.2682515741720801, 0.2573637364478126, 0.2770957707973042]

def get_train_loader(args, dataset_class, use_sobel=False, use_color=False):
    # Data loading code
    normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                                     std=CIFAR100_TRAIN_STD)
    img_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = dataset_class(f'{args.img_dir}/train', transform=img_transform, use_sobel=use_sobel, use_color=use_color)

    train_dataloader = data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True)

    return train_dataloader

def get_val_loader(args, dataset_class):
    # Data loading code
    normalize = transforms.Normalize(mean=CIFAR100_TRAIN_MEAN,
                                     std=CIFAR100_TRAIN_STD)
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = dataset_class(f'{args.img_dir}/test', transform=img_transform)

    train_dataloader = data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True)

    return train_dataloader

