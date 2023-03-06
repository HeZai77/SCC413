import glob
import random
import os
import cv2
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, phase="train"):
        self.transform = transforms_
        if phase == "train":
            self.files_A = sorted(glob.glob(os.path.join(root, 'trainA') + '/*.jpg'))
            self.files_B = sorted(glob.glob(os.path.join(root, 'trainB') + '/*.jpg'))
        elif phase == "test":
            self.files_A = sorted(glob.glob(os.path.join(root, 'testA') + '/*.jpg'))
            self.files_B = sorted(glob.glob(os.path.join(root, 'testB') + '/*.jpg'))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index]).convert('RGB')
        img_B = Image.open(self.files_B[index]).convert('RGB')
        # img_A = np.array(img_A) / 255.0
        # img_B = np.array(img_B) / 255.0

        item_A = self.transform(img_A)
        item_B = self.transform(img_B)

        return item_A, item_B

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


class SubImageDataset(Dataset):
    def __init__(self, root, transforms_=None, dataname="A"):
        self.transform = transforms_
        self.files = sorted(glob.glob(root + '/*.jpg'))
        self.dataname = dataname

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        if self.dataname == "A":
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')

        # img.show()
        # img = np.array(img) / 255.0
        # img = img.reshape(512, 512, 1)[:, :, 0]
        # img = Image.fromarray(img)
        item = self.transform(img)
        return item

    def __len__(self):
        return len(self.files)


class SubImageDataset_v2(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms_
        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        img_file = self.files[index]
        if img_file.endswith('raw'):
            imgData = np.fromfile(img_file, dtype=np.float32)
            imgData = imgData.reshape(512, 512, 1)[:, :, 0]
            img = Image.fromarray((imgData*255).astype(np.uint8))

        elif img_file.endswith('dcm'):
            imgData = dcmread(img_file)
            imgData = imgData.pixel_array
            imgData = imgData.astype(np.float32)
            imgData = np.reshape(imgData, (768, 1024, 1))[:, :, 0]
            imgDataNorm = (imgData - np.min(imgData)) / (np.max(imgData) - np.min(imgData))
            img_pil = Image.fromarray((imgDataNorm * 255).astype(np.uint8))
            img = img_pil.resize((512, 512))

        else:
            img = Image.open(img_file)
        # img = img.convert('L')
        # img.show()
        # img = np.array(img) / 255.0
        # img = img.reshape(512, 512, 1)[:, :, 0]
        # img = Image.fromarray(img)
        item = self.transform(img)
        return item

    def __len__(self):
        return len(self.files)



class UnalignedDataset(Dataset):
    def __init__(self, root, transforms_=None, mode='train'):
        self.transform = transforms_
        self.files_A = sorted(glob.glob(os.path.join(root, 'trainA') + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, 'trainB') + '/*.*'))

    def __getitem__(self, index):
        file_A = self.files_A[index]
        index_B = random.randint(0, len(self.files_B)-1)
        file_B = self.files_B[index_B]
        img_A = Image.open(file_A).convert('L')
        img_B = Image.open(file_B).convert('L')
        img_A = np.array(img_A) / 255.0
        img_B = np.array(img_B) / 255.0

        item_A = self.transform(Image.fromarray(img_A))
        item_B = self.transform(Image.fromarray(img_B))

        return {'A': item_A, 'B': item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))