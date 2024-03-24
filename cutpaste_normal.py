#coding=gbk
import random
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import os
import cv2
import imgaug.augmenters as iaa
from perlin import rand_perlin_2d_np
from  utils import CutPasteNormal,qianjing
from utils import Scar_test

class CutPastePlus2(object):
    def __init__(self, **kwags):
        self.normal = CutPasteNormal(**kwags)
        self.ROTE = False
        self.img_size_O = self.normal.img_size_O
        self.resize_shape = [self.img_size_O, self.img_size_O]
        root_dtd = f'C:/datasets/DATA_SETS/dtd/images'
        files = os.listdir(f"{root_dtd}")
        self.anomaly_source_paths = []
        for file in files:
            all_images = os.listdir(f"{root_dtd}/{file}")
            for sig_img in all_images:
                self.anomaly_source_paths.append(f"{root_dtd}/{file}/{sig_img}")
        self.after_cutpaste_transform = transforms.Compose([])
        self.after_cutpaste_transform.transforms.append(transforms.ToTensor())
        self.after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.data_type = self.normal.data_type
        self.use_wenli_only = self.normal.use_wenli_only
        self.use_jiegou_only = self.normal.use_jiegou_only
        self.without_qianjing = self.normal.without_qianjing
        self.model_name = self.normal.model_name
        self.CUTPASTE = self.normal.CUTPASTE
        self.all_train_imgs=self.normal.all_train_imgs
        self.epoch_MODE=self.normal.epoch_MODE
        self.train_imgs_len = len(self.all_train_imgs)#.shape[0]
        print('use_jiegou_only:', self.use_jiegou_only)
        print('use_wenli_only:', self.use_wenli_only)
        print('without_qianjing:', self.without_qianjing)
        print('CUTPASTE:', self.CUTPASTE)
        print('train_imgs_len:', self.train_imgs_len)
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True), iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(), iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)), iaa.Posterize(), iaa.Invert(), iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(), iaa.Affine(rotate=(-45, 45))]
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.rot_scar = iaa.Sequential([iaa.Affine(rotate=(-5, 5))])

    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]], self.augmenters[aug_ind[1]], self.augmenters[aug_ind[2]]])
        return aug

    def augment_image(self, image, anomaly_source_path, scar=None, without_qianjing=False):
        aug = self.randAugmenter()
        min_perlin_scale = 0
        if scar:
            perlin_scale = 3
            threshold = 0.4
        else:
            perlin_scale = 4
            threshold = 0.5
        if scar:
            anomaly_source_img = image.copy()
            anomaly_img_augmented = Scar_test(anomaly_source_img,self.data_type,self.train_imgs_len,self.all_train_imgs)
        else:
            anomaly_source_img = cv2.imread(anomaly_source_path)
            try:
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
            except:
                print(anomaly_source_path)
            anomaly_img_augmented = aug(image=anomaly_source_img)

        new_perlin_thr = np.zeros((self.img_size_O, self.img_size_O))
        num = 0
        TRY_TIME = 0
        PISH_PINS = 36 if self.data_type == 'pushpins' else 20
        TH = True
        SUM_BL = 64 if self.data_type=='pushpins' else 49
        while new_perlin_thr.sum() < SUM_BL:
            num += 1
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
            while (perlin_scalex == 64) and (perlin_scaley == 64):
                perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
                perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])

            perlin_noise = rand_perlin_2d_np((self.resize_shape[0], self.resize_shape[1]),(perlin_scalex, perlin_scaley))
            perlin_noise = self.rot(image=perlin_noise)
            perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
            perlin_thr = np.expand_dims(perlin_thr, axis=2)
            new_perlin_thr, TH, PISH_PINS, TRY_TIME = qianjing(self.data_type,perlin_thr,image,TH,scar,anomaly_img_augmented,PISH_PINS,TRY_TIME)

        perlin_thr = new_perlin_thr
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0
        if scar:
            beta = torch.rand(1).numpy()[0] * 0.25
        else:
            beta = 0.75 if torch.rand(1).numpy()[0] > 0.75 else torch.rand(1).numpy()[0] * 0.9
        if self.data_type == 'splicing_connectors':
            beta = 0 if scar else beta
        if self.data_type == 'pushpins':
            beta = 0 if scar else torch.rand(1).numpy()[0] * 0.4
        if self.data_type == 'juice_bottle':
            beta = 0
        if self.data_type == 'breakfast_box':
            beta = torch.rand(1).numpy()[0] * 0.7 if ~scar else beta
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)  # perlin_thr为0 1 掩码    img_thr为将要叠加的噪声图像
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1 - msk) * image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly = 0.0
        msk = cv2.resize(msk,(256,256))
        msk[msk >= 0.5] = 1
        msk[msk < 0.5] = 0
        return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)

    def transform_image(self, image, anomaly_source_path,scar = None,without_qianjing=False):
        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(
            np.float32) / 255.0  # 归一化图像
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path,scar = scar,without_qianjing = without_qianjing)
        augmented_image = (augmented_image * 255).astype(np.uint8)
        image = (image * 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_RGB2BGR)
        return image, augmented_image, anomaly_mask, has_anomaly

    def __call__(self, img):
        if  self.data_type == 'breakfast_box':
            r = random.uniform(0, 0.7)#*20
        else:
            r = random.uniform(0, 0.6)
        SCAR = 1 if r < 0.35 else 0

        img_org = img
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths),(1,)).item()  # anomaly_source_paths是所有datasets\dtd\images下的图片
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(img, self.anomaly_source_paths[anomaly_source_idx],scar = SCAR,without_qianjing=self.without_qianjing) # 传入图片的路径
        anomaly_mask = torch.tensor(anomaly_mask)
        anomaly_mask = anomaly_mask.unsqueeze(2)
        anomaly_mask = anomaly_mask.unsqueeze(0)
        mask_cut = anomaly_mask.squeeze(3)

        augmented_image = Image.fromarray(augmented_image)
        augmented_cut = self.after_cutpaste_transform(augmented_image)

        img_org = self.after_cutpaste_transform(img_org)
        return img_org, augmented_cut, mask_cut
