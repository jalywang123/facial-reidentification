import os
import PIL
import torch
import random
import numpy as np
from PIL import Image
from torch import nn as nn
from utils import get_face_locations

# from skimage import io, transform
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        # we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0, 1)
        if should_get_same_class:
            while True:
                # keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if (img0_tuple[1] == img1_tuple[1]) and (
                    img0_tuple[0] != img1_tuple[0]
                ):
                    break
        else:
            while True:
                # keep looping till a different class image is found

                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0 = img0_tuple[0]
        #img0 = get_face_locations(img0)
        img0 = Image.open(img0)

        img1 = img1_tuple[0]
        #img1 = get_face_locations(img1)
        img1 = Image.open(img1)

        img0 = img0.convert("L")
        img1 = img1.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)

        return (
            img0 / 255.0,
            img1 / 255.0,
            torch.from_numpy(
                np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32)
            ),
        )

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class TripletDataset(Dataset):
    def __init__(self, imageFolderDataset, transform=None, should_invert=True):
        self.imageFolderDataset = imageFolderDataset
        self.transform = transform
        self.should_invert = should_invert

    def __getitem__(self, index):
        # getting anchor image
        anchor = random.choice(self.imageFolderDataset.imgs)
        # we need to get one image of same class and on of a different class
        random.randint(0, 1)
        while True:

            # keep looping till the same class image is found
            positive = random.choice(self.imageFolderDataset.imgs)
            if (anchor[1] == positive[1]) and (anchor[0] != positive[0]):
                break
        while True:
            # keep looping till a different class image is found

            negative = random.choice(self.imageFolderDataset.imgs)
            if anchor[1] != negative[1]:
                break

        img0 = anchor[0]
        img0 = get_face_locations(img0)

        img1 = positive[0]
        img1 = get_face_locations(img1)

        img2 = negative[0]
        img2 = get_face_locations(img2)

        img0 = img0.convert("L")
        img1 = img1.convert("L")
        img2 = img2.convert("L")

        if self.should_invert:
            img0 = PIL.ImageOps.invert(img0)
            img1 = PIL.ImageOps.invert(img1)
            img2 = PIL.ImageOps.invert(img2)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img0, img1, img2

    def __len__(self):
        return len(self.imageFolderDataset.imgs)


class ImagesDataset(Dataset):
    def __init__(self, rootdir, transform=None):
        self.rootdir = rootdir
        self.transform = transform

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = f"{os.listdir(self.rootdir)[idx]}"
        #img = get_face_locations(f"{self.rootdir}/{img_name}")
        img = Image.open(f"{self.rootdir}/{img_name}")
        img = img.convert("L")
        # img = np.asarray(image)
        # img = img/255.0
        # img = torch.from_numpy(np.asarray(img))

        if self.transform is not None:
            img = self.transform(img)

        sample = {"image": img / 255.0, "name": img_name}

        return sample

    def __len__(self):
        return len(os.listdir(self.rootdir))


class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(3, padding=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(3, padding=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(3, padding=1),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
        )  # 10-float32 bit encoding

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2

class SiameseNetwork_V2(nn.Module):
    def __init__(self):
        super(SiameseNetwork_V2, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            
            nn.MaxPool2d(3, padding=1),
            
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(16),
            
            nn.MaxPool2d(3, padding=1),

#             nn.ReflectionPad2d(1),
#             nn.Conv2d(8, 8, kernel_size=3),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(8),
            
#             nn.MaxPool2d(3, padding=1),

        )

        self.fc1 = nn.Sequential(
            nn.Linear(2304, 500),
            nn.ReLU(inplace=True),

#             nn.Linear(500, 500),
#             nn.ReLU(inplace=True),

            nn.Linear(500, 128)) # 10-float32 bit encoding

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2
