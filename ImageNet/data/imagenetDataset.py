from torch.utils import data
import glob
from PIL import Image
import numpy as np
import random
import torch
import os
import cv2

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

def pil_to_sobel(image):
    gray = image.convert("L")
    np_gray = np.array(gray, dtype=np.uint8)

    sobelx = cv2.Sobel(np_gray, cv2.CV_64F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(np_gray, cv2.CV_64F, 0, 1, ksize=5)  # y
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = (sobel / sobel.max() * 255)

    pil_sobel = Image.fromarray(sobel).convert("RGB")
    pil_sobel = Image.blend(pil_sobel, image, 0.15)

    return pil_sobel

def pil_to_colored(image):
    return image.resize([16, 16]).resize([224, 224])

class imagenetDataset(data.Dataset):
    def __init__(self, img_dir, preprocess=None, transform=None, target_transform=None, loader=default_loader, use_sobel=False, use_color=False):
        self.images = []
        self.targets = []
        self.class_str_to_id = {}
        self.class_id_to_str = {}

        self.loader = loader
        self.transform = transform
        self.preprocess = preprocess
        self.target_transform = target_transform
        self.use_sobel = use_sobel
        self.use_color = use_color

        img_classes = glob.glob(f'{img_dir}/*')

        for idx, img_class_path in enumerate(img_classes):
            img_class = img_class_path.split('/')[-1]
            self.class_str_to_id[img_class] = idx
            self.class_id_to_str[idx] = img_class

            imgs_pathes = glob.glob(f'{img_dir}/{img_class}/*.JPEG')

            for img_path in imgs_pathes:
                self.images.append(img_path)
                self.targets.append(idx)

    def __getitem__(self, index):
        img_path = self.images[index]
        sample = self.loader(img_path)
        extra_data = {}

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        if self.use_color:
            colorized = pil_to_colored(sample)
        if self.use_sobel:
            sobel = pil_to_sobel(sample)

        target = self.targets[index]

        if self.transform is not None:
            sample = self.transform(sample)

            if self.use_sobel:
                sobel = self.transform(sobel)
                extra_data['sobel'] = sobel

            if self.use_color:
                colorized = self.transform(colorized)
                extra_data['color'] = colorized

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, extra_data

    def __len__(self):
        return len(self.images)
