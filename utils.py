#coding=gbk
import random
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import cv2
from PIL import  ImageFilter
import albumentations


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')



def cut_paste_collate_fn(batch):
    img_types = list(zip(*batch))
    return [torch.stack(imgs) for imgs in img_types]



class CutPaste(object):
    def __init__(self, colorJitter=0.1, transform=None,args = None,data_type = None,use_jiegou_only=False,use_wenli_only=False,
                 without_qianjing = False,model_name = 'r',CUTPASTE=0,all_train_imgs=None,img_size_O=None,epoch_MODE=0):
        self.transform = transform
        self.args = args
        self.data_type = data_type
        self.use_jiegou_only = use_jiegou_only
        self.use_wenli_only = use_wenli_only
        self.without_qianjing = without_qianjing
        self.model_name = model_name
        self.CUTPASTE = CUTPASTE
        self.all_train_imgs=all_train_imgs
        self.img_size_O = img_size_O
        self.epoch_MODE=epoch_MODE
        print('CutPaste')
        colorJitter = True
        if colorJitter is None:
            self.colorJitter = None
        else:
            self.colorJitter = transforms.ColorJitter(brightness = 0.1,saturation = 0.1,hue = 0.1)
    def __call__(self, org_img, img ,mask):
        pass


class CutPasteNormal(CutPaste):
    def __init__(self, area_ratio=[0.02,0.15], aspect_ratio=0.3, **kwags):
        super(CutPasteNormal, self).__init__(**kwags)
        self.ROTE = False
    def __call__(self, img):
        pass


def Scar_test(anomaly_source_img,data_type,train_imgs_len,all_train_imgs):
    f = torch.rand(1).numpy()[0]
    if f >= 0.5:
        anomaly_source_img = cv2.flip(anomaly_source_img, -1)
    temp = (anomaly_source_img * 255).astype(np.uint8)

    Flip = random.uniform(0, 1)
    if Flip > 0.3:
        a_t = transforms.ColorJitter(brightness=0.4, saturation=0.4, hue=0.2)
    else:
        a_t = transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1)

    if data_type == 'juice_bottle':
        a_t = transforms.ColorJitter(brightness=0.2, saturation=0.2, hue=0.1)
    patch = Image.fromarray(cv2.cvtColor(temp, cv2.COLOR_BGR2RGB))
    Flip = random.uniform(0, 1)
    if data_type == 'splicing_connectors' or data_type == 'juice_bottle' or data_type == 'breakfast_box':
        pass
    else:
        if Flip > 0.8:
            patch = a_t(patch)
        elif data_type == 'pushpins' and Flip > 0.3:
            patch = a_t(patch)
        elif data_type == 'screw_bag' and Flip > 0.0:
            a_t = transforms.ColorJitter(brightness=0.7, saturation=0.7, hue=0.45)
            patch = a_t(patch)
        Flur = random.uniform(0, 1)
        if Flur > 0.8:
            patch = patch.filter(ImageFilter.GaussianBlur(radius=2))

    temp = cv2.cvtColor(np.asarray(patch), cv2.COLOR_RGB2BGR)
    anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(4, 8), always_apply=True)(image=temp)['image']

    if data_type == 'juice_bottle':
        Flur = random.uniform(0, 1)
        if Flur > 0.75:
            anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(3, 3), always_apply=True)(image=temp)['image']
        elif Flur < 0.333:
            anomaly_img_augmented = albumentations.vflip(temp)
        else:
            INDEX_SE = np.random.randint(0, train_imgs_len, 1)
            anomaly_img_augmented = np.array(all_train_imgs[INDEX_SE[0]])
            anomaly_img_augmented = cv2.cvtColor(np.array(anomaly_img_augmented), cv2.COLOR_RGB2BGR)

    if data_type == 'breakfast_box':
        Flur = random.uniform(0, 1)
        if Flur > 0.666:
            anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(3, 2), always_apply=True)(image=temp)['image']
        elif Flur < 0.333:
            anomaly_img_augmented = albumentations.vflip(temp)
        else:
            anomaly_img_augmented = albumentations.hflip(temp)

    if data_type == 'pushpins':
        Flur = random.uniform(0, 1)
        if Flur > 0.75:
            anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(3, 5), always_apply=True)(image=temp)['image']
        elif Flur < 0.333:
            anomaly_img_augmented = albumentations.vflip(temp)
        else:
            INDEX_SE = np.random.randint(0, train_imgs_len, 1)
            anomaly_img_augmented = np.array(all_train_imgs[INDEX_SE[0]])
            anomaly_img_augmented = cv2.cvtColor(np.array(anomaly_img_augmented), cv2.COLOR_RGB2BGR)

    if data_type == 'screw_bag':
        Flur = random.uniform(0, 1)
        if Flur > 0.75:
            anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(4, 4), always_apply=True)(image=temp)['image']
        elif Flur < 0.25:
            Flur_ = random.uniform(0, 1)
            if Flur_ >= 0.5:
                anomaly_img_augmented = albumentations.vflip(temp)
            else:
                anomaly_img_augmented = albumentations.hflip(temp)
        else:
            INDEX_SE = np.random.randint(0, train_imgs_len, 1)
            anomaly_img_augmented = np.array(all_train_imgs[INDEX_SE[0]])
            anomaly_img_augmented = cv2.cvtColor(np.array(anomaly_img_augmented), cv2.COLOR_RGB2BGR)
            anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(4, 4), always_apply=True)(image=anomaly_img_augmented)['image']

    if data_type == 'splicing_connectors':
        Flur = random.uniform(0, 1)
        THUID = 0.8
        if Flur > 0.3:  # 0.7
            INDEX_SE = np.random.randint(0, train_imgs_len, 1)
            anomaly_img_augmented = np.array(all_train_imgs[INDEX_SE[0]])
            anomaly_img_augmented = cv2.cvtColor(np.array(anomaly_img_augmented), cv2.COLOR_RGB2BGR)
            Flurss = random.uniform(0, 1)
            if Flurss <= (THUID - 0.05):
                anomaly_img_augmented[0:64, :, :] = anomaly_img_augmented[64:128, :, :]
                anomaly_img_augmented[192:256, :, :] = anomaly_img_augmented[128:192, :, :]
                anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(1, 4), always_apply=True)(image=anomaly_img_augmented)['image']
        else:
            Flurss = random.uniform(0, 1)
            if Flurss <= THUID:
                temp[0:64, :, :] = temp[64:128, :, :]
                temp[192:256, :, :] = temp[128:192, :, :]
            anomaly_img_augmented = albumentations.RandomGridShuffle(grid=(2, 4), always_apply=True)(image=temp)['image']
    return  anomaly_img_augmented



def qianjing(data_type,perlin_thr,image,TH,scar,anomaly_img_augmented,PISH_PINS,TRY_TIME):
    PRO_FORE = random.uniform(0, 1)
    THTHTH = 0.25  # all_test_th[self.epoch_MODE]
    if data_type == "pushpins" and TH == True and PRO_FORE > THTHTH and scar:
        ret, image_cv = cv2.threshold(anomaly_img_augmented, PISH_PINS, 255, cv2.THRESH_BINARY)
        image_cv = image_cv[:, :, 0]
        kerne2 = np.ones((5, 5), np.uint8)
        image_cv = cv2.dilate(image_cv, kerne2, iterations=1)
        image_cv = np.expand_dims(image_cv / 255.0, axis=2)
        perlin_thr = perlin_thr * image_cv
        TRY_TIME += 1
        if TRY_TIME % 3 == 0:
            PISH_PINS -= 1
            if PISH_PINS == 1:
                PISH_PINS += 1
        if TRY_TIME >= 16:
            TH = False
    if data_type == "splicing_connectors" and TH == True and PRO_FORE > 0.8 and scar:
        ret, image_cv = cv2.threshold(anomaly_img_augmented, 20, 255, cv2.THRESH_BINARY)
        image_cv = image_cv[:, :, 0]

        kerne2 = np.ones((6, 13), np.uint8)
        image_cv = cv2.erode(image_cv, kerne2, iterations=1)
        kerne2 = np.ones((9, 15), np.uint8)
        image_cv = cv2.dilate(image_cv, kerne2, iterations=1)

        image_cv = np.expand_dims(image_cv / 255.0, axis=2)
        perlin_thr = image_cv
        TRY_TIME += 1
        if TRY_TIME % 3 == 0:
            PISH_PINS -= 1
            if PISH_PINS == 1:
                PISH_PINS += 1
        if TRY_TIME >= 16:
            TH = False

    if data_type == "splicing_connectors":
        new_perlin_thr = perlin_thr

    elif data_type == "juice_bottle":
        image_cv = (image * 255).astype(np.uint8)
        ret, image_cv = cv2.threshold(image_cv, 8, 255, cv2.THRESH_BINARY)
        image_cv = image_cv[:, :, 0]
        kerne2 = np.ones((15, 15), np.uint8)
        image_cv = cv2.erode(image_cv, kerne2, iterations=1)
        kerne2 = np.ones((3, 3), np.uint8)
        image_cv = cv2.dilate(image_cv, kerne2, iterations=1)
        image_cv = np.expand_dims(image_cv / 255.0, axis=2)
        new_perlin_thr = perlin_thr * image_cv
        new_perlin_thr[:, 201:, :] = 0
        new_perlin_thr[240:, :, :] = 0

    elif data_type == "breakfast_box":
        new_perlin_thr = perlin_thr
        Flur = random.uniform(0, 1)
        if Flur > 0.8:  # 0.7
            new_perlin_thr[140:205, 140:205, :] = 1
    else:
        new_perlin_thr = perlin_thr
    return new_perlin_thr,TH,PISH_PINS,TRY_TIME
