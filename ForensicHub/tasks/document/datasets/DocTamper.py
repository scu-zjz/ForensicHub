import os
import cv2
import six
import math
import lmdb
import torch
import random
import jpegio
import pickle
import argparse
import tempfile
import numpy as np
from torch import nn
from PIL import Image
from tqdm import tqdm
import albumentations as A
import torch.optim as optim
from torchvision.transforms import v2
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union, no_type_check
from ForensicHub.registry import register_dataset
from ForensicHub.core.base_dataset import BaseDataset


@register_dataset("DocTamperData")
class DocTamperData(BaseDataset):
    def __init__(self, path, train=True, crop_size=512, jpeg=True, dct_path=None, crop_test=False, suffix_img='.jpg', suffix_mask='.png',**kwargs):
        self.jpeg = True # must use jpeg augmentation
        self.train = train # train or test
        self.dct_path = dct_path # return dct or not
        self.crop_size = crop_size # model input size
        self.crop_test = False
        self.path = path
        self.suffix_img = suffix_img
        self.suffix_mask = suffix_mask
        if dct_path:
            # other_files/qt_table_ori.pk
            with open(dct_path, 'rb') as f:
                self.qtables = pickle.load(f)
        super().__init__(path=path, **kwargs)
        print(path, self.__len__, 'train:', train)


    def _init_dataset_path(self):
        self.images = [(os.path.join(self.path, 'images', x), os.path.join(self.path, 'masks', x[:-len(self.suffix_img)]+self.suffix_mask)) for x in os.listdir(os.path.join(self.path, 'images')) if x.endswith(self.suffix_img)]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        image = Image.open(img_path)
        mask = np.clip(cv2.imread(mask_path, 0), 0, 1)
        h,w = image.size
        if self.train: # random-resize + crop
            if self.dct_path:
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    image.save(tmp,"JPEG",quality=random.randint(75, 100))
                    jpg = jpegio.read(tmp.name)
                    dct = jpg.coef_arrays[0].copy()
                    qtb = jpg.quant_tables[0].copy()
                    if qtb.max()>61:
                        print("WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!")
                        qtb = self.qtables[75]
                    image = Image.open(tmp.name)
        else:
            if self.dct_path:
                jpg = jpegio.read(img_path)
                dct = jpg.coef_arrays[0].copy()
                qtb = jpg.quant_tables[0].copy()
                if qtb.max()>61:
                    print("WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!")
                    qtb = self.qtables[75]
        image = np.array(image)
        # img = self.totsr(img)
        if self.common_transform:
            output = self.common_transform(image=image, mask=mask)
            image = output['image']
            mask = output['mask']
        mask = torch.LongTensor(mask)
        mask = mask.unsqueeze(0)
        label = (mask.sum(dim=(0, 1, 2)) != 0).long() 
        if self.post_transform:
            image = self.post_transform(image=image)['image']
        if self.dct_path:
            return {'image': image, 'mask': mask, 'label':label, 'dct': np.clip(np.abs(dct), 0,20), 'qtb': qtb}
        else:
            return {'image': image, 'mask': mask, 'label':label}

if __name__=='__main__':
    data_names = (('/mnt/data0/public_datasets/Doc/DocTamperV1/DocTamperV1-TrainingSet', False), ('/mnt/data0/public_datasets/Doc/DocTamperV1/DocTamperV1-TrainingSet', True))
    for v in data_names:
        data = DocTamperData(path=v[0], train=v[1])
        for i in range(10):
            item = data.__getitem__(0)
            img = item['image']
            mask = item['mask']
        import pdb;pdb.set_trace()
            # if use_dct:
            #     dct = item['dct']
            #     qtb = item['qtb']
            #     print(data_names, i, img.shape, mask.shape, dct.shape, qtb.shape)
            # else:
            #     print(data_names, i, img.shape, mask.shape)
             
            
