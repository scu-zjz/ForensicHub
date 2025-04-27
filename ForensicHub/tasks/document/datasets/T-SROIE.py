import os
import cv2
import six
import math
import lmdb
import timm
import torch
import pickle
import random
import jpegio
import argparse
import tempfile
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm
from PIL import Image
from copy import deepcopy
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from crop_comb import crop_img_func
from typing import List, Optional, Tuple, Union, no_type_check
from ForensicHub.registry import register_dataset

@register_dataset("DocTamperData")
class SROIEData(Dataset):
    def __init__(self, root, train=True, crop_size=512, jpeg=True, use_dct=False, crop_test=True, suffix_img='.jpg', suffix_mask='.png'):
        self.train = train # train or test
        self.use_dct = use_dct # return dct or not
        self.crop_size = crop_size # model input size
        self.crop_test = (crop_test and (not train))
        self.images = [(os.path.join(root, 'images', x), os.path.join(root, 'masks', x[:-len(suffix_img)]+suffix_mask)) for x in os.listdir(os.path.join(root, 'images')) if x.endswith(suffix_img)]
        self.lens = len(self.images)
        print(root, self.lens, 'train:', train)
        if train:
            with open('other_files/train.pk','rb') as fpk: # 训练文件路径
                self.fpk = pickle.load(fpk)
        with open('other_files/qt_table.pk','rb') as fpk:
            pks = pickle.load(fpk)
        self.pks = {}
        for k,v in pks.items():
            self.pks[k] = torch.Tensor(v)
        with open('other_files/qtm75.pk','rb') as fpk:
            self.pkf = pickle.load(fpk)
            self.pkfkeys = self.pkf.keys()
        self.qs = np.arange(90,101)
        self.hflip = torchvision.transforms.RandomHorizontalFlip(p=1.0)
        self.vflip = torchvision.transforms.RandomVerticalFlip(p=1.0)
        self.totsr = A.ToTensorV2()
        self.totsr2 = A.Compose([
            A.Resize(height=crop_size, width=crop_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ])
        self.toctsr =torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=(0.485, 0.455, 0.406), std=(0.229, 0.224, 0.225))])

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        image = Image.open(img_path)
        mask = np.clip(cv2.imread(mask_path, 0), 0, 1)
        if not self.train:
            img = np.array(image)
            if self.use_dct:
                jpg = jpegio.read(img_path)
                dct = jpg.coef_arrays[0].copy()
                qtb = jpg.quant_tables[0].copy()
                assert qtb.max()<=61, "WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!"
                imgs, meta_info, dcts = crop_img_func(img, img_name_ori=img_path.split('/')[-1], mask=None, jpg_dct=dct, crop_size=self.crop_size)
                for k in dcts.keys():
                    dcts[k] = torch.LongTensor(np.clip(np.abs(dcts[k]),0,20))
                for k in imgs.keys():
                    imgs[k] = self.totsr2(image=imgs[k])['image']
                return {'img': imgs, 'mask': torch.LongTensor(mask), 'dct': dcts, 'qtb': qtb.reshape(8,8)}
            else:
                imgs, meta_info = crop_img_func(img, img_name_ori=img_path.split('/')[-1], mask=None, jpg_dct=None, crop_size=self.crop_size)
                for k in imgs.keys():
                    imgs[k] = self.totsr2(image=imgs[k])['image']
                return {'img': imgs, 'mask': torch.LongTensor(mask)}

        imgi = img_path.split('/')[-1]
        H,W = mask.shape[:2]
        if H < 512:
            dh = (512-H)
        else:
            dh = 0
        if W < 512:
            dw = (512-W)
        else:
            dw = 0
        mask = np.pad(mask,((0,dh),(0,dw)),'constant',constant_values=0)
        image = np.pad(image,((0,dh),(0,dw),(0,0)),'constant',constant_values=255)
        H,W = mask.shape
        H864 = H//8-64
        W864 = W//8-64
        if random.uniform(0,1)<0.1:
          if (len(self.fpk[imgi])!=0):
            sxu,syu = random.choice(self.fpk[imgi])
          else:
            H,W = mask.shape[:2]
            sxu = random.randint(0,H864)
            syu = random.randint(0,W864)
        elif random.uniform(0,1)<0.2:
          for t in range(4):
            sxu = random.randint(0,H864)*8
            syu = random.randint(0,W864)*8
            mask_ = mask[sxu:(sxu+512),syu:(syu+512)]
            if mask_.max() != 0:
                break
        else:
            sxu = random.randint(0,H864)*8
            syu = random.randint(0,W864)*8
            if random.uniform(0,1)<0.15:
                sxu = 0
            if random.uniform(0,1)<0.15:
                syu = 0
        image = image[sxu:sxu+512,syu:syu+512]
        mask = mask[sxu:sxu+512,syu:syu+512]
        if (self.train and (random.uniform(0,1)<0.3)):
            qu = random.randint(2,12)
            while not (qu in self.pkfkeys):
                qu = random.randint(2,12)
            qu2 = qu+random.randint(1,24)
            while not (qu2 in self.pkfkeys):
                qu2 = qu+random.randint(1,24)
            image = Image.fromarray(image).convert('L')
            image2 = deepcopy(image)
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                image.save(tmp,"JPEG",qtables={0:random.choice(self.pkf[qu])})
                image = np.array(Image.open(tmp))
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                image2.save(tmp,"JPEG",qtables={0:random.choice(self.pkf[qu2])})
                image2 = np.array(Image.open(tmp))
            im = Image.fromarray(image*mask+image2*(1-mask)).convert('RGB')
        else:
            im = Image.fromarray(image)
        masksum = mask.sum()
        if masksum<8:
            cls_lbl = 0
        elif masksum>512:
            cls_lbl = 1
        else:
            cls_lbl = -1
        mask = self.totsr(image=mask.copy())['image']
        q1,q2 = np.random.choice(self.qs,2,replace=False)
        q1 = int(q1)
        q2 = int(q2)
        use_qt = torch.stack((self.pks[q1],self.pks[q2]))
        sidx_temp = torch.stack([torch.randperm(2) for i in range(64)],1).reshape(2,8,8)
        new_qt = [nqt.short().flatten().tolist() for nqt in torch.gather(use_qt,0,sidx_temp)]
        new_qtb = {0:[int(x) for x in new_qt[0]]}
        if random.uniform(0,1) < 0.5:
            im = self.hflip(im)
            mask = self.hflip(mask)
        if random.uniform(0,1) < 0.5:
            im = self.vflip(im)
            mask = self.vflip(mask)
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            im = im.convert("L")
            im.save(tmp,"JPEG",qtables=new_qtb)
            jpg = jpegio.read(tmp.name)
            dct = jpg.coef_arrays[0].copy()
            im = im.convert('RGB')
        return {'img': self.toctsr(im), 'mask': mask.long(), 'dct': np.clip(np.abs(dct),0,20), 'qtb': torch.LongTensor(new_qt[0])}
        
if __name__=='__main__':
    for use_dct in (True, False):
        data_names = (('sroie_train', False), ('sroie_train', True))
        for v in data_names:
            print('AAA', v, use_dct)
            data = SROIEData(root='../'+v[0], train=v[1], use_dct=use_dct)
            for i in range(10):
                item = data.__getitem__(0)
                img = item['img']
                mask = item['mask']
                if use_dct:
                    dct = item['dct']
                    qtb = item['qtb']
                    if v[1]:
                        print(data_names, i, img.shape, mask.shape)
                    else:
                        print(data_names, i, [x.shape for x in img.values()], mask.shape, [x.shape for x in dct.values()], qtb.shape)
                else:
                    if v[1]:
                        print(data_names, i, img.shape, mask.shape)
                    else:
                        print(data_names, i, [x.shape for x in img.values()], mask.shape, qtb.shape)
            
