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
from IMDLBenCo.transforms import EdgeMaskGenerator
from IMDLBenCo.model_zoo.mvss_net.mvssnet import MVSSNet


# register_dataset('DocTamperDataOrigin')
class DocTamperDataOrigin(Dataset):
    def __init__(self, root, train=True, crop_size=512, jpeg=True, use_dct=False, crop_test=False, suffix_img='.jpg', suffix_mask='.png',edge_width=None):
        self.jpeg = True # must use jpeg augmentation
        self.train = train # train or test
        self.use_dct = use_dct # return dct or not
        self.crop_size = crop_size # model input size
        self.crop_test = False
        self.images = [(os.path.join(root, 'images', x), os.path.join(root, 'masks', x[:-len(suffix_img)]+suffix_mask)) for x in os.listdir(os.path.join(root, 'images')) if x.endswith(suffix_img)]
        self.lens = len(self.images)
        print(root, self.lens, 'train:', train)
        self.totsr = v2.Compose([
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        if use_dct:
            with open('/mnt/data0/public_datasets/Doc/DataLoaders/other_files/qt_table_ori.pk', 'rb') as f:
                self.qtables = pickle.load(f)
        self.edge_mask_generator = None if edge_width is None else EdgeMaskGenerator(edge_width)
    def __len__(self):
        return self.lens

    def __getitem__(self, index):
        img_path, mask_path = self.images[index]
        img = Image.open(img_path)
        mask = np.clip(cv2.imread(mask_path, 0), 0, 1)
        h,w = img.size
        if self.train: # random-resize + crop
            if self.use_dct:
                with tempfile.NamedTemporaryFile(delete=True) as tmp:
                    img.save(tmp,"JPEG",quality=random.randint(75, 100))
                    jpg = jpegio.read(tmp.name)
                    dct = jpg.coef_arrays[0].copy()
                    qtb = jpg.quant_tables[0].copy()
                    if qtb.max()>61:
                        print("WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!")
                        qtb = self.qtables[75]
                    img = Image.open(tmp.name)
        else:
            if self.use_dct:
                jpg = jpegio.read(img_path)
                dct = jpg.coef_arrays[0].copy()
                qtb = jpg.quant_tables[0].copy()
                if qtb.max()>61:
                    print("WARNING! The image quality is too low, exceeding the model's capabilities. This will lead to bad results!")
                    qtb = self.qtables[75]
        img = self.totsr(img)
        mask = torch.LongTensor(mask).unsqueeze(0).float()
        label = (torch.sum(mask, dim=(0, 1, 2)) != 0).long()
        
        if self.use_dct:
            data_dict = {'image': img, 'mask': mask, 'label':label, 'DCT_coef': torch.tensor(np.clip(np.abs(dct),0,20)), 'qtables': torch.tensor(qtb).unsqueeze(0)}
        else:
            data_dict = {'image': img, 'mask': mask, 'label':label}

        if self.edge_mask_generator != None: 
            gt_img_edge = self.edge_mask_generator(mask)[0]
            data_dict['edge_mask'] = torch.tensor(gt_img_edge).float()
        return data_dict
    
if __name__=='__main__':
    data_names = (('/mnt/data0/public_datasets/Doc/DocTamperV1/DocTamperV1-TrainingSet', False), ('DocTamperV1-TrainingSet', True))
    for use_dct in (True, False):
        for v in data_names:
            data = DocTamperDataOrigin(root=v[0], train=v[1], use_dct=use_dct, edge_width=7)
            for i in range(10):
                item = data.__getitem__(i)
                img = item['image']
                mask = item['mask']
                if use_dct:
                    dct = item['DCT_coef']
                    qtb = item['qtables']
                    print(data_names, i, img.shape, mask.shape, dct.shape, qtb.shape)
                else:
                    print(data_names, i, img.shape, mask.shape)
                

                import pdb;pdb.set_trace()
                # item = {k:v.unsqueeze(0) for k, v in item.items()}
                # for k,v in item.items():
                #     v = v.unsqueeze(0)
                # model = MVSSNet()
