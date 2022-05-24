"""A modified image folder class

We modify the official PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
so that this class can load images from both current directory and its subdirectories.
"""

import torch.utils.data as data
from data.base_dataset import get_transformSL

from PIL import Image
import numpy as np
import os
import natsort

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images[:min(max_dataset_size, len(images))]


def default_loader(path):
    return Image.open(path).convert('RGB')

def my_loader(path):
    i = Image.open(path)
    return np.array(i).astype(float)

class SLDataset(data.Dataset):

    def __init__(self,opt, rootInput,rootOutput, return_paths=False,
                 loader=default_loader):
        inputs = make_dataset(rootInput)
        outputs = make_dataset(rootOutput)
        
        self.inputsRoot = rootInput
        self.outputsRoot = rootOutput
        self.inputs = natsort.natsorted(inputs)
        self.outputs = natsort.natsorted(outputs)
        self.loader = loader
        self.transforms = get_transformSL(opt, grayscale=True)

    def __getitem__(self, index):
        pathI = self.inputs[index]
        pathO = self.outputs[index]
        i = self.loader(pathI)
        o = self.loader(pathO)

        if self.transforms is not None:

            i = self.transforms(i)
            o = self.transforms(o)

        # if self.return_paths:
        #     return img, path
        # else:
        return i,o

    def __len__(self):
        return len(self.inputs)


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)