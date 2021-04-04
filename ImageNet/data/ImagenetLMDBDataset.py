import os
import os.path as osp
from PIL import Image
import six
import lmdb
import pyarrow as pa
import numpy as np
import cv2

import torch.utils.data as data

def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)

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


def pil_to_blur(image):
    np_img = np.array(image, dtype=np.uint8)
    blur = cv2.GaussianBlur(np_img, (51, 51), 0)
    pil_blur = Image.fromarray(blur).convert("RGB")

    return pil_blur

class ImageFolderLMDB(data.Dataset):
    def __init__(self, db_path, preprocess=None, transform=None, target_transform=None, use_sobel=False, use_color=False):
        self.db_path = db_path
        self.env = lmdb.open(db_path, subdir=osp.isdir(db_path),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

        self.preprocess = preprocess
        self.transform = transform
        self.target_transform = target_transform
        self.use_sobel = use_sobel
        self.use_color = use_color

    def __getitem__(self, index):
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        unpacked = loads_pyarrow(byteflow)

        # load img
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        sample = Image.open(buf).convert('RGB')

        # load label
        target = unpacked[1]

        extra_data = {}

        if self.preprocess is not None:
            sample = self.preprocess(sample)

        if self.use_color:
            colorized = pil_to_colored(sample)
        if self.use_sobel:
            sobel = pil_to_sobel(sample)

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
        return self.length

    def __repr__(self):
        return self.__class__.__name__ + ' (' + self.db_path + ')'