import numpy as np
import torch
import cv2
from train import get_mse, get_peak_points
from models import KFSGNet
from dataGen import ALFWDataset, save_face_with_kps, ConvertTensor
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def visualize_heatmap_target(oriImg, heatmap, idx=100):
    return 0
    # plt.imshow(oriImg, cmap=plt.cm.gray)
    # plt.imshow(heatmap, alpha=.5)
    # plt.savefig('heatmaps_{}.jpg'.format(idx))


def get_peak_points(heatmaps):
    """
    get keypoints by heatmaps
    :param heatmaps: numpy array (N,21,96,96)
    :return:numpy array (N,21,2)
    """
    C,H,W = heatmaps.shape
    all_peak_points = []
    peak_points = []
    for j in range(C):
        yy,xx = np.where(heatmaps[j] == heatmaps[j].max())
        y = yy[0]
        x = xx[0]
        peak_points.append([x,y])
    return np.array(peak_points)


def save_face_with_kps(image, kps, idx=0):

    for points in kps:
        if points[0] > 0:
            cv2.circle(image, (points[0], points[1]), 2, (0, 255, 0))
    cv2.imwrite('imgkps_{}.jpg'.format(idx), image)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = ConvertTensor()
dataset = ALFWDataset('face_test_with_rect.csv', transform=transform)
dataset.load()

batch_size = 1
dataloader = DataLoader(dataset, batch_size, shuffle=True)

hourglass = KFSGNet()
hourglass.load_state_dict(torch.load('kd_epoch_99_model.ckpt'))
hourglass.float().to(device)
hourglass.eval()

sample = next(iter(dataloader))
inputs = sample['image']
inputs = inputs.to(device)
image_target = sample['image'].to('cpu').numpy().squeeze()
heatmap_target = sample['heatmaps'].to('cpu').numpy().squeeze()

output = hourglass(inputs)
inputs = inputs.to('cpu').numpy().squeeze()
output = output.to('cpu').data.numpy().squeeze()
all_landmarks = get_peak_points(output)
print(all_landmarks)
save_face_with_kps(image_target*255, all_landmarks, idx=100)
for i, heatmap in enumerate(output):
    visualize_heatmap_target(inputs, heatmap, idx=i)
    # visualize_heatmap_target(inputs, heatmap_target[i])
    max_y, max_x = np.where(heatmap == heatmap.max())
    label_x, label_y = sample['gts'].to('cpu').numpy().squeeze()[i]
    print("max_x: {} : {}, max_y: {} : {}".format(max_x[0], label_x, max_y[0], label_y))

# print(output)


