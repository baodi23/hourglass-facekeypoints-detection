import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd 
import numpy as np 
import ast
import cv2
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class RandomFlip(object):

    def __init__(self):
        super().__init__()

    def __call__(self, sample):
        sample['image'] = transforms.RandomVerticalFlip()(sample['image'])
        return sample

class Rescale(object):

    def __init__(self, outsize):
        assert isinstance(outsize, (int, tuple))
        self.__outsize = outsize

    def __call__(self, sample):
        image, gts = sample['image'], sample['gts']
        h, w = image.shape[:2]
        # print("h: {} w: {}".format(h, w))
        if isinstance(self.__outsize, int):
            if h > w:
                new_h, new_w = self.__outsize * h / w, self.__outsize
            else:
                new_h, new_w = self.__outsize, self.__outsize * w / h 
        else:
            new_h, new_w = self.__outsize

        new_h, new_w = int(new_h), int(new_w)
        image = Image.fromarray(image, mode='L')
        image = image.resize((new_w, new_h), Image.ANTIALIAS)
        image = np.array(image)

        for index, point in enumerate(gts):
            if point[0] >= 0:
                gts[index] *= [new_w / w, new_h / h]

        gts = gts.astype(np.int32)
        return {'image': image,
                'gts': gts,
                'rect': sample['rect']}


class ConvertTensor(object):

    def __call__(self, sample):
        image, gts, rect, heatmaps = sample['image'], sample['gts'], sample['rect'], sample['heatmaps']

        # numpy image H x W x C
        # torch image C x H x W
        image = np.reshape(image, (image.shape[0], image.shape[1], 1))
        image = image.transpose((2, 0, 1))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image)
        gts = torch.from_numpy(gts)
        rect = np.array(rect)
        rect = torch.from_numpy(rect)
        if heatmaps is not None:
            heatmaps = torch.from_numpy(heatmaps)
        return {'image': image,
                'gts': gts,
                'heatmaps': heatmaps}


class ALFWDataset(Dataset):

    def __init__(self, file_csv, is_test=False, transform=None, is_visible=False):

        self.__fname = file_csv
        self.__is_test = is_test
        self.__transform = transform
        self.__imgpaths = []
        self.__gts = []
        self.__rects = []
        self.__is_visible = is_visible

    def load(self):

        file_df = pd.read_csv(self.__fname)
        for index, row in file_df.iterrows():
            imgpath = row['file_path']
            gts = row['face_coords']
            rect = row['face_rect']
            self.__imgpaths.append(imgpath)
            self.__gts.append(gts)
            self.__rects.append(rect)

    def __getitem__(self, index):

        rescale_y, rescale_x = 96, 96
        image_path = self.__imgpaths[index]
        img = Image.open(image_path).convert('L')
        img = np.array(img)
        img_h, img_w = img.shape[:2]
        # img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        gts_dir = self.__gts[index]
        gts_dir = ast.literal_eval(gts_dir)
        face_rect = self.__rects[index]
        face_rect = ast.literal_eval(face_rect)
        face_rect[0] -= face_rect[2] * 0.05
        face_rect[1] -= face_rect[3] * 0.05
        face_rect[2] *= 1.1
        face_rect[3] *= 1.1
        # crop face image
        if face_rect[0] < 0:
            face_rect[2] += face_rect[0]
            face_rect[0] = 0
        if face_rect[1] < 0:
            face_rect[3] += face_rect[1]
            face_rect[1] = 0
        if face_rect[0]+face_rect[2] > img_w:
            face_rect[2] = img_w - face_rect[0]
        if face_rect[1]+face_rect[3] > img_h:
            face_rect[3] = img_h - face_rect[1]
        rect_xright = int(face_rect[0] + face_rect[2])
        rect_yright = int(face_rect[1] + face_rect[3])
        # print("rect: {}, img: {} {} {} {}".format(face_rect, img_w, img_h, rect_xright, rect_yright))
        # print(image_path)
        img = img[int(face_rect[1]):rect_yright, int(face_rect[0]):rect_xright]

        kps = []
        coordinates = []

        # get feature id and feature coordinates
        for key in range(1, 22):
            kps.append(key)
            if key in gts_dir.keys():
                gts_dir[key][0] -= face_rect[0]
                gts_dir[key][1] -= face_rect[1]
                coordinates.append(gts_dir[key][0])
                coordinates.append(gts_dir[key][1])
            else:
                coordinates.append(-1.0)
                coordinates.append(-1.0)

        coordinates = np.array(coordinates).astype(np.float32).reshape((-1, 2))
        sample = {'image': img, 'gts': coordinates, 'rect': face_rect}
        sample = Rescale((rescale_x, rescale_y))(sample)

        if self.__is_test:
            sample['heatmaps'] = None
        else:
            heatmaps = self._putGaussianMaps(sample['gts'], rescale_y, rescale_x)
            sample['heatmaps'] = heatmaps

        if self.__is_visible:
            self.visualize_heatmaps_target(np.squeeze(sample['image']), heatmaps)

        if self.__transform is not None:
            sample = self.__transform(sample)
        else:
            sample = ConvertTensor()(sample)

        return sample

    def __len__(self):
        return len(self.__imgpaths)

    def _putGaussianMap(self, center, visible_flag, crop_size_y, crop_size_x, stride=1, sigma=5.):
        """
        根据一个中心点生成 heatmap
        :param center: 关键点坐标
        :param visible_flag: 关键点是否可见
        :param crop_size_y: 图片高
        :param crop_size_x: 图片宽
        :param stride:
        :param sigma: 高斯函数参数
        :return: heatmap crop_size_y x crop_size_x
        """
        grid_y = int(crop_size_y / stride)
        grid_x = int(crop_size_x / stride)
        if not visible_flag:
            return np.zeros((grid_y, grid_x))
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def _putGaussianMaps(self, centers, crop_size_y, crop_size_x, stride=1, sigma=5.):
        """
        根据所有关键点生成一组 heatmaps
        :param centers: 关键点坐标
        :param crop_size_y: 图片高
        :param crop_size_x: 图片宽
        :param stride:
        :param sigma: 高斯函数参数
        :return: heatmaps 21 x crop_size_y x crop_size_x
        """
        heatmaps_this_face = []
        visible_flag = False
        for center in centers:
            if center[0] > 0:
                visible_flag = True
            else:
                visible_flag = False
            heatmap = self._putGaussianMap(center, visible_flag, crop_size_y, crop_size_x, stride, sigma)
            heatmap = heatmap[np.newaxis, ...]
            heatmaps_this_face.append(heatmap)
        heatmaps_this_face = np.concatenate(heatmaps_this_face, axis=0)
        return heatmaps_this_face

    def visualize_heatmap_target(self, oriImg, heatmap):

        plt.imshow(oriImg, cmap=plt.cm.gray)
        plt.imshow(heatmap, alpha=.5)
        plt.show()

    def visualize_heatmaps_target(self, oriImg, heatmaps):

        for i, heatmap in enumerate(heatmaps):
            # plt.subplot(7, 3, i+1)
            plt.imshow(oriImg, cmap=plt.cm.gray)
            plt.imshow(heatmap, alpha=.5)
            plt.show()


def save_face_with_kps(image, kps, idx=0):

    for points in kps:
        if points[0] > 0:
            cv2.circle(image, (points[0], points[1]), 2, (0, 255, 0))
    cv2.imwrite('imgkps_{}.jpg'.format(idx), image)


if __name__ == "__main__":
    
    composed = transforms.Compose([ConvertTensor(),
                                   RandomFlip()])
    dataset = ALFWDataset('./face_list_df_with_rect.csv', transform=ConvertTensor(), is_visible=False)
    dataset.load()
    dataloader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    for index, sample in enumerate(dataloader):
        img = sample['image'][0].numpy().transpose((1, 2, 0))
        img *= 255
        img = np.squeeze(img)
        # img = Image.fromarray(img, mode='L')
        # img = np.array(img)
        save_face_with_kps(img, sample['gts'][0].numpy(), index)

        if index > 12:
            break
