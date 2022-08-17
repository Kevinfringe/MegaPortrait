import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F
import model
import HeadPoseEstimation
from torch.utils.data import DataLoader
import dataset

source_img_path = './source_img'
driver_img_path = './driver_img'
data_set_length = 42

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

input1 = torch.randn(2, 3, 256, 256)


def tensor2image(image, norm=255.0):
    return (image.squeeze().permute(0, 1, 2).numpy() * norm)


# mean = input1.mean(dim=0, keepdim=True)
#
# test_model = model.Emtn_facial(in_channels=3)
# test_model2 = model.Eapp2(repeat=[3, 4, 6, 3])
# output = test_model(input1)
# print(output)
# output2 = test_model2(input1)
# print(output2)
# sum = output2 + output
# print(sum)
#
# print("The dimension of the sum is: " + str(sum.size()))
# test_model = model.Eapp1()
# output = test_model(input1)

# Test for dataloader
source_set_origin = dataset.SourceDataset(source_img_path, data_set_length, transform=None)
driver_set_origin = dataset.DriverDataset(driver_img_path, data_set_length, transform=None)

source_loader_origin = DataLoader(source_set_origin, batch_size=2, shuffle=False)
driver_loader_origin = DataLoader(driver_set_origin, batch_size=2, shuffle=False)

imgs = next(iter(source_loader_origin))
print("imgs.shape is :" + str(imgs.size()))
w_rt = HeadPoseEstimation.head_pose_estimation(imgs)

#print(output)

print(w_rt.numpy().shape)

#Test for warp_generator
# input_warp = torch.randn(2, 1,  256)
# warp_generator = model.WarpGenerator(input_channels=1)
# output_warp = warp_generator(input_warp)
#
# print(output_warp)

# Test for head-pose removal.


for i in range(2):
    print(imgs[i].shape)
    img = tensor2image(imgs[i]).astype(np.uint8)
    print("********** image *********")
    print(img.shape)
    homography = w_rt[i].numpy()
    print("********* homography ********")
    print(homography.shape)
    output = cv.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

    plt.imshow(output)
    plt.show()
