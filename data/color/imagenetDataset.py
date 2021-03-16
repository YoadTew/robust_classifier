from torch.utils import data
import glob
from PIL import Image
import numpy as np
import random
import torch
import os

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    return pil_loader(path)

def colroize_loader(path):
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    else:
        return Image.fromarray(np.zeros((4, 4))).convert('RGB')

class imagenetDataset(data.Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None, loader=default_loader):
        self.images = []
        self.colorized_imgs = []
        self.targets = []
        self.class_str_to_id = {}
        self.class_id_to_str = {}

        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        img_classes = glob.glob(f'{img_dir}/*')

        for idx, img_class_path in enumerate(img_classes):
            img_class = img_class_path.split('/')[-1]
            self.class_str_to_id[img_class] = idx
            self.class_id_to_str[idx] = img_class

            imgs_pathes = glob.glob(f'{img_dir}/{img_class}/*.JPEG')

            for img_path in imgs_pathes:
                img_name = img_path.split('/')[-1]

                self.images.append(img_path)
                self.colorized_imgs.append(f'{img_dir}/{img_class}/colorize/{img_name}')
                self.targets.append(idx)

    def __getitem__(self, index):
        img_path, colorized_path = self.images[index], self.colorized_imgs[index]
        sample = self.loader(img_path)
        colorized = colroize_loader(colorized_path)

        target = self.targets[index]
        # target = torch.LongTensor(target)

        if self.transform is not None:

            # Need this for colorized transform consistency
            seed = np.random.randint(2147483647)

            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            sample = self.transform(sample)

            random.seed(seed)  # apply this seed to img tranfsorms
            torch.manual_seed(seed)  # needed for torchvision 0.7
            colorized = self.transform(colorized)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, colorized

    def __len__(self):
        return len(self.images)
