from torch.utils import data
import torchvision.transforms as transforms
import os
import torchvision

import glob

from data.AugMixDataset import AugMixDataset


def get_train_loader(args, dataset_class, use_sobel=False, use_color=False):
    # Data loading code
    img_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
        # transforms.RandomRotation(15),
    ])

    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])


    dataset = dataset_class(f'{args.img_dir}/train', transform=img_transform, use_sobel=use_sobel, use_color=use_color)
    dataset = AugMixDataset(dataset, preprocess, args, args.no_jsd)

    train_dataloader = data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True)

    return train_dataloader

def get_val_loader(args, dataset_class):
    # Data loading code
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    dataset = dataset_class(f'{args.img_dir}/test', transform=img_transform)

    train_dataloader = data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=True,
                                       drop_last=True, pin_memory=True)

    return train_dataloader

