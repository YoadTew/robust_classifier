from torch.utils import data
import torchvision.transforms as transforms
from data.imagenetCorruptedDataset import imagenetCorruptedDataset
import os

import glob

def get_train_loader(args, dataset_class, use_sobel=False, use_color=False):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip()
    ])

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = dataset_class(f'{args.img_dir}/train', preprocess=preprocess, transform=img_transform, use_sobel=use_sobel, use_color=use_color)

    train_dataloader = data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return train_dataloader

def get_val_loader(args, dataset_class):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    dataset = dataset_class(f'{args.img_dir}/val', transform=img_transform)

    train_dataloader = data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size, shuffle=True)

    return train_dataloader

def get_test_loader(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    dataset = imagenetCorruptedDataset(f'{args.img_dir}', transform=img_transform)

    train_dataloader = data.DataLoader(dataset, num_workers=args.n_workers, batch_size=args.batch_size)

    return train_dataloader

def get_test_loaders(args):

    img_dir = args.img_dir
    parent_augs = ['blur', 'weather', 'noise', 'digital']

    for parent_aug in parent_augs:
        aug_childs = [x.split('/')[-1] for x in glob.glob(f'{img_dir}/{parent_aug}/*')]

        for aug in aug_childs:
            for aug_level in range(1,6):
                args.img_dir = f'{img_dir}/{parent_aug}/{aug}/{aug_level}'

                yield get_test_loader(args), f'{aug}_{aug_level} ({parent_aug})'
