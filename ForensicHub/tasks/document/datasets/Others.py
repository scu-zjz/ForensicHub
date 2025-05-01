import os
import cv2
import six
import math
import timm
import torch
import random
import jpegio
import pickle
import argparse
import tempfile
import numpy as np
from PIL import Image
from torch import nn
from tqdm import tqdm
import albumentations as A
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
# from crop_comb import crop_img_func
from typing import List, Optional, Tuple, Union, no_type_check
from ForensicHub.registry import register_dataset
from IMDLBenCo.transforms import EdgeMaskGenerator

# @register_dataset("OthersData")
class OthersData(Dataset):
    def __init__(self, root, train=True, crop_size=512, jpeg=True, dct_path='/mnt/data0/public_datasets/Doc/DataLoaders/other_files/qt_table_ori.pk', crop_test=True, suffix_img='.jpg', suffix_mask='.png', edge_width=7):
        self.jpeg = jpeg # use jpeg augmentation
        self.train = train # train or test
        self.dct_path = dct_path # return dct or not
        self.crop_size = crop_size # model input size
        self.crop_test = (crop_test and (not train))
        self.images = [(os.path.join(root, 'images', x), os.path.join(root, 'masks', x[:-len(suffix_img)]+suffix_mask)) for x in os.listdir(os.path.join(root, 'images')) if x.endswith(suffix_img)]
        self.lens = len(self.images)
        print(root, self.lens, 'train:', train)
        self.crops = A.Compose([
            A.RandomCrop(height=crop_size, width=crop_size, pad_if_needed=True, pad_position="top_left", border_mode=cv2.BORDER_CONSTANT, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.geometric.RandomRotate90(),
            A.RGBShift(p=1.0),
        ])
        self.totsr = A.Compose([
            A.Resize(height=crop_size, width=crop_size, interpolation=cv2.INTER_LINEAR),
            A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            A.ToTensorV2(),
        ])
        if dct_path:
            # /mnt/data1/public_datasets/Doc/DataLoaders/other_files/qt_table_ori.pk'
            with open(dct_path, 'rb') as f:
                self.qtables = pickle.load(f)
        # if crop_test:
        #     assert not train, 'crop test is only True in test mode.'
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)

    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        img = cv2.imread(img_path)
        mask = np.clip(cv2.imread(mask_path, 0), 0, 1)
        h,w = img.shape[:2]
        if self.train: # random-resize + crop
            if random.uniform(0,1)<0.5: # keep_ratio
                new_ratio = random.uniform(0.5, 2.0)
                new_h = int(h * new_ratio + 0.5)
                new_w = int(w * new_ratio + 0.5)
                img = cv2.resize(img, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h))
            else:
                new_hr = random.uniform(0.75, 1.33)
                new_wr = random.uniform(0.75, 1.33)
                new_h = int(h * new_hr + 0.5)
                new_w = int(w * new_wr + 0.5)
                img = cv2.resize(img, (new_w, new_h))
                mask = cv2.resize(mask, (new_w, new_h))
            cropped = self.crops(image = img, mask = mask)
            img, mask = cropped['image'], cropped['mask']
            if self.jpeg:
                if self.dct_path:
                    im = Image.fromarray(img)
                    with tempfile.NamedTemporaryFile(delete=True) as tmp:
                        im = im.convert("L")
                        im.save(tmp,"JPEG",quality=random.randint(75, 100))
                        jpg = jpegio.read(tmp.name)
                        dct = jpg.coef_arrays[0].copy()
                        qtb = jpg.quant_tables[0].copy()
                        if qtb.max()>61:
                            print("WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!")
                            qtb = self.qtables[75]
                img = cv2.imdecode(cv2.imencode('.jpg', img, (1, random.randint(75, 100)))[1], 1)
            else:
                if self.dct_path:
                    jpg = jpegio.read(img_path)
                    dct = jpg.coef_arrays[0].copy()
                    qtb = jpg.quant_tables[0].copy()
                    if qtb.max()>61:
                        print("WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!")
                        qtb = self.qtables[75]
        else:
            if self.dct_path:
                jpg = jpegio.read(img_path)
                dct = jpg.coef_arrays[0].copy()
                qtb = jpg.quant_tables[0].copy()
                if qtb.max()>61:
                    print("WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!")
                    qtb = self.qtables[75]
    
        if not self.crop_test:
            if not self.train:
                img = cv2.resize(img, (self.crop_size, self.crop_size))
            img = self.totsr(image=img)['image']
            mask = torch.LongTensor(mask).unsqueeze(0).float()
            label = (torch.sum(mask, dim=(0, 1, 2)) != 0).long()
            if self.dct_path:
                data_dict = {'image': img, 'mask': mask, 'label':label, 'DCT_coef': torch.tensor(np.clip(np.abs(dct),0,20)), 'qtables': torch.tensor(qtb)}
            else:
                data_dict = {'image': img, 'mask': mask, 'label':label}
        else:
            if self.dct_path:
                imgs, meta_info, dcts = crop_img_func(img, img_name_ori=img_path.split('/')[-1], mask=None, jpg_dct=dct, crop_size=self.crop_size)
                for k in dcts.keys():
                    dcts[k] = torch.LongTensor(np.clip(np.abs(dcts[k]),0,20)).contiguous()
                for k in imgs.keys():
                    imgs[k] = self.totsr(image=imgs[k])['image']
                data_dict = {'image': img, 'mask': mask, 'label':label, 'DCT_coef': dcts, 'qtables': torch.tensor(qtb).unsqueeze(0)}
            else:
                imgs, meta_info = crop_img_func(img, img_name_ori=img_path.split('/')[-1], mask=None, jpg_dct=None, crop_size=self.crop_size)
                for k in imgs.keys():
                    imgs[k] = self.totsr(image=imgs[k])['image']
                data_dict = {'image': img, 'mask': mask, 'label':label}
            
        if self.edge_mask_generator != None: 
            gt_img_edge = self.edge_mask_generator(mask)[0]
            data_dict['edge_mask'] = torch.tensor(gt_img_edge).float()
        return data_dict
        
if __name__=='__main__':
    train_dataset = OthersData('/mnt/data0/public_datasets/Doc/RealTextManipulation/RTM_train', train=True, edge_width=7)
    train_dataset.__getitem__(0)
    import pdb;pdb.set_trace()