from torch.utils.data import Dataset
import numpy as np
import random
import cv2
import os
import spectral
import matplotlib.pyplot as plt

class TrainDataset(Dataset):
    def __init__(self, data_root='dataset/pear dataset/train_dataset/', crop_size=128, arg=True, bgr2rgb=True, stride=128):
        self.crop_size = crop_size 
        self.hypers = [] #cropped hsi data
        self.bgrs = []   # cropped rgb data
        self.arg = arg # augment
        self.h, self.w = 512,512  # img shape
        self.stride = stride
        self.patch_per_line = (self.w-crop_size)//self.stride + 1
        self.patch_per_colum = (self.h-crop_size)//self.stride + 1
        self.patch_per_img = self.patch_per_line * self.patch_per_colum

        dataset_list = os.listdir(data_root)

        for image_folder in dataset_list:
            if image_folder == '.DS_Store' or image_folder == 'Readme.md':
                continue
            file_path = data_root+image_folder
            rgb_path = file_path + '/' + image_folder + '.png'
            hsi_path = file_path + '/REFLECTANCE_' + image_folder + '.hdr'
            print(image_folder, " loaded")

            bgr = cv2.imread(rgb_path)
            if True:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
            bgr = np.float32(bgr)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,512,512]
            self.bgrs.append(bgr)

            hyper = spectral.open_image(hsi_path).load()
            hyper = np.transpose(hyper, (2, 1, 0)) # [204, 512, 512] 
            self.hypers.append(hyper)

        self.img_num = len(self.hypers)
        self.length = self.patch_per_img * self.img_num

    def arguement(self, img, rotTimes, vFlip, hFlip):
        # Random rotation
        for j in range(rotTimes):
            img = np.rot90(img.copy(), axes=(1, 2))
        # Random vertical Flip
        for j in range(vFlip):
            img = img[:, :, ::-1].copy()
        # Random horizontal Flip
        for j in range(hFlip):
            img = img[:, ::-1, :].copy()
        return img

    def __getitem__(self, idx):
        stride = self.stride
        crop_size = self.crop_size
        img_idx, patch_idx = idx//self.patch_per_img, idx%self.patch_per_img
        h_idx, w_idx = patch_idx//self.patch_per_line, patch_idx%self.patch_per_line
        bgr = self.bgrs[img_idx]
        hyper = self.hypers[img_idx]
        bgr = bgr[:,h_idx*stride:h_idx*stride+crop_size, w_idx*stride:w_idx*stride+crop_size]
        hyper = hyper[:, h_idx * stride:h_idx * stride + crop_size,w_idx * stride:w_idx * stride + crop_size]
        
        rotTimes = random.randint(0, 3)
        vFlip = random.randint(0, 1)
        hFlip = random.randint(0, 1)
        if self.arg:
            bgr = self.arguement(bgr, rotTimes, vFlip, hFlip)
            hyper = self.arguement(hyper, rotTimes, vFlip, hFlip)
        
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return self.patch_per_img * self.img_num



class ValidDataset(Dataset):
    def __init__(self, data_root='dataset/pear dataset/val_dataset/', bgr2rgb=True):
        self.hypers = [] #cropped hsi data
        self.bgrs = []   # cropped rgb data

        dataset_list = os.listdir(data_root)
        print(' Validation Dataset')

        for image_folder in dataset_list:
            if image_folder == '.DS_Store' or image_folder == 'Readme.md':
                continue
            file_path = data_root+image_folder
            rgb_path = file_path + '/' + image_folder + '.png'
            hsi_path = file_path + '/REFLECTANCE_' + image_folder + '.hdr'
            print(image_folder, " loaded")

            bgr = cv2.imread(rgb_path)
            if True:
                bgr = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                
            bgr = np.float32(bgr)
            bgr = (bgr-bgr.min())/(bgr.max()-bgr.min())
            bgr = np.transpose(bgr, [2, 0, 1])  # [3,512,512]
            self.bgrs.append(bgr)

            hyper = spectral.open_image(hsi_path).load()
            hyper = np.transpose(hyper, (2, 1, 0)) # [204, 512, 512] 
            self.hypers.append(hyper)


    def __getitem__(self, idx):
        hyper = self.hypers[idx]
        bgr = self.bgrs[idx]
        return np.ascontiguousarray(bgr), np.ascontiguousarray(hyper)

    def __len__(self):
        return len(self.hypers)

if __name__ == "__main__:
    train_data = TrainDataset()
    len(train_data)
    
    val_dataset = ValidDataset()
    len(val_dataset)
