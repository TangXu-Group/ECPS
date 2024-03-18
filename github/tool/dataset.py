import os
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np
import torch
from torchvision import transforms as tvtsf
import random
import cv2
import torchvision.transforms.functional as tf
from PIL import Image
import PIL

def my_transform0(image1, image2, label):
    # 随机变色
    RanJ = tvtsf.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
    image1, image2 = RanJ(image1), RanJ(image2)
    return image1, image2, label

def my_transform1(image1, image2, label):
    # 拿到角度的随机数。angle是一个-10到10之间的一个数
    angle = tvtsf.RandomRotation.get_params([-10, 10])
    # 对image和mask做相同的旋转操作，保证他们都旋转angle角度
    image1 = image1.rotate(angle)
    image2 = image2.rotate(angle)
    label = label.rotate(angle)
    return image1, image2, label

def my_transform2(image1, image2, label):
    # 50%的概率应用垂直，水平翻转。
    if random.random() > 0.5:
        image1 = tf.hflip(image1)
        image2 = tf.hflip(image2)
        label = tf.hflip(label)
    if random.random() > 0.5:
        image1 = tf.vflip(image1)
        image2 = tf.vflip(image2)
        label = tf.vflip(label)
    return image1, image2, label

def my_transform3(image1, image2, label):
    # 随机裁剪
    i, j, h, w = tvtsf.RandomResizedCrop.get_params(
    image1, scale=(0.7, 1.0), ratio=(1, 1))
    size = (image1.width, image1.height)
    
    image1 = tf.resized_crop(image1, i, j, h, w, size)
    image2 = tf.resized_crop(image2, i, j, h, w, size)
    label = tf.resized_crop(label, i, j, h, w, size, interpolation=PIL.Image.NEAREST)
    return image1, image2, label

def tranform_sum(image1, image2, label):
#     image1, image2, label = my_transform0(image1, image2, label)
    image1, image2, label = my_transform1(image1, image2, label)
    image1, image2, label = my_transform2(image1, image2, label)
    image1, image2, label = my_transform3(image1, image2, label)
    return image1, image2, label

# General
def pytorch_normalzeA(img):
    normalize = tvtsf.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[1,1,1])
    img = normalize(torch.from_numpy(img).float())
    return img.numpy()
def pytorch_normalzeB(img):
    normalize = tvtsf.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[1,1,1])
    img = normalize(torch.from_numpy(img).float())
    return img.numpy()

class Dataset(Dataset):

    def __init__(self,img_path,label_path,file_name_txt_path,split_flag, transform=True):

        self.label_path = label_path
        self.img_path = img_path
        self.img_txt_path = file_name_txt_path
        self.imgs_path_list = np.loadtxt(self.img_txt_path,dtype=str)
        self.flag = split_flag
        self.transform = transform
        self.img_label_path_pairs = self.get_img_label_path_pairs()

    def get_img_label_path_pairs(self):

        img_label_pair_list = {}
        if self.flag =='train':
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name,image2_name,mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path, mask_name)
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,image2_name])

        if self.flag == 'val' or self.flag == 'test':
            for idx , did in enumerate(open(self.img_txt_path)):
                try:
                    image1_name, image2_name, mask_name = did.strip("\n").split(' ')
                except ValueError:  # Adhoc for test.
                    image_name = mask_name = did.strip("\n")
                extract_name = image1_name[image1_name.rindex('/') +1: image1_name.rindex('.')]
                img1_file = os.path.join(self.img_path , image1_name)
                img2_file = os.path.join(self.img_path , image2_name)
                lbl_file = os.path.join(self.label_path , mask_name)
                img_label_pair_list.setdefault(idx, [img1_file,img2_file,lbl_file,image2_name])

        return img_label_pair_list

    def data_transform(self, img1, img2, lbl):
        img1 = img1[:, :, ::-1]  # RGB -> BGR
        img1 = img1.astype(np.float64).transpose(2, 0, 1)
        img1 /= 255
        img1 = pytorch_normalzeA(img1)

        img2 = img2[:, :, ::-1]  # RGB -> BGR
        img2 = img2.astype(np.float64).transpose(2, 0, 1)
        img2 /= 255
        img2 = pytorch_normalzeB(img2)
        
        lbl = (torch.tensor(lbl)>100).int()
        return img1,img2,lbl

    def __getitem__(self, index):

        img1_path,img2_path,label_path,filename = self.img_label_path_pairs[index]
        ####### load images and label #############
        img1 = Image.open(img1_path) #.resize(size=(512,512),resample=PIL.Image.NEAREST)
        img2 = Image.open(img2_path) #.resize(size=(512,512),resample=PIL.Image.NEAREST)
        label = Image.open(label_path) #.resize(size=(512,512),resample=PIL.Image.NEAREST)
        if len(np.array(label).shape)==3:
            label = np.array(label)
            label = label[:,:,0]
            label=Image.fromarray(label)
        
        height,width = img1.height, img1.width
        
        if self.transform == True:
            img1, img2, label = tranform_sum(img1, img2, label)

        img1 = np.array(img1)
        img2 = np.array(img2)
        label = np.array(label)
        train_mask = (np.sum(img1,axis=2)!=0).astype(int)
        
        img1, img2, label = self.data_transform(img1, img2, label)
        
        
        return img1,img2,label,str(filename),train_mask #,int(height),int(width)

    def __len__(self):

        return len(self.img_label_path_pairs)