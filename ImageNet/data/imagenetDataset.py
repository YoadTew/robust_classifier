from torch.utils import data
import glob
from PIL import Image
import numpy as np
import random
import torch
import os
import cv2
import json
import time

import torchvision.datasets as datasets
import torchvision.transforms as transforms

to_tensor = transforms.ToTensor()

def pil_to_sobel_old(image):
    gray = image.convert("L")
    np_gray = np.array(gray, dtype=np.uint8)

    sobelx = cv2.Sobel(np_gray, cv2.CV_64F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(np_gray, cv2.CV_64F, 0, 1, ksize=5)  # y
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = (sobel / sobel.max() * 255)

    pil_sobel = Image.fromarray(sobel).convert("RGB")
    pil_sobel = Image.blend(pil_sobel, image, 0.15)

    return pil_sobel

def pil_to_colored_old(image):
    return image.resize([16, 16]).resize([224, 224])

def pil_to_blur(image):
    np_img = np.array(image, dtype=np.uint8)
    blur = cv2.GaussianBlur(np_img, (51, 51), 0)
    pil_blur = Image.fromarray(blur).convert("RGB")

    return pil_blur

def pil_to_sobel(image):
    gray = image.convert("L")

    np_gray = np.array(gray, dtype=np.uint8)

    sobelx = cv2.Sobel(np_gray, cv2.CV_32F, 1, 0, ksize=5)  # x
    sobely = cv2.Sobel(np_gray, cv2.CV_32F, 0, 1, ksize=5)  # y
    sobel = np.sqrt(sobelx ** 2 + sobely ** 2)
    sobel = (sobel / sobel.max())

    sobel = to_tensor(sobel).repeat(3, 1, 1)

    return sobel

pil_to_colored = transforms.Compose([
        transforms.Resize([16, 16]),
        transforms.Resize([224, 224]),
        to_tensor
    ])


class imagenetDataset(data.Dataset):
    def __init__(self, img_dir, preprocess=None, transform=None, target_transform=None, use_sobel=False, use_color=False):

        self.dataset = datasets.ImageFolder(img_dir, preprocess)

        self.transform = transform
        self.target_transform = target_transform
        self.use_sobel = use_sobel
        self.use_color = use_color

    def __getitem__(self, index):
        sample, target = self.dataset[index]

        extra_data = {}

        # Sobel and colorize receive PIL and return tensor
        if self.use_color:
            colorized = pil_to_colored(sample)
            colorized = self.transform(colorized)
            extra_data['color'] = colorized

        if self.use_sobel:
            sobel = pil_to_sobel(sample)
            sobel = self.transform(sobel)
            extra_data['sobel'] = sobel

        sample = to_tensor(sample)
        sample = self.transform(sample)

        return sample, target, extra_data

    def __len__(self):
        return len(self.dataset)
