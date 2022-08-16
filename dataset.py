import numpy as np
from torch.utils.data import Dataset
import cv2 as cv

source_img_path = './source_img'
driver_img_path = './driver_img'

class SourceDataset(Dataset):
    def __init__(self, source_img_path, length,transform=None):
        self.source_img_path = source_img_path
        self.length = length
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.source_img_path + "A" + str(item) + ".jpg"
        img = cv.imread(img_path)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        return img


class DriverDataset(Dataset):
    def __init__(self, driver_img_path, length, transform=None):
        self.driver_img_path = driver_img_path
        self.length = length
        self.transform = transform

    def __getitem__(self, item):
        img_path = self.driver_img_path + "G" + str(item) + ".jpg"
        img = cv.imread(img_path)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        return img