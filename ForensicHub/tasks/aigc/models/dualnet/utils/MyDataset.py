import torch
import torchvision.datasets as datasets
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
#import PIL.Image as Image
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
import os
import imutils

class MyDataset(Dataset):
    def __init__(self, txt, trans=None, option=None, type=None, size=None, augment=None):
        self.root = txt
        assert(type!=None)
        assert(size!=None)
        self.option = option
        if type == "DALLE":
            if option is not None:
                self.Neg_root = os.path.join("/data/xiziyi/DALLE/DA_"+ str(size) + "_JPG_postpro/DALLE/", option) + '/'
                self.Pos_root = os.path.join("/data/xiziyi/DALLE/DA_" + str(size) + "_JPG_postpro/ALASKA/", option) + '/'
            else:
                self.Neg_root = "/data/xiziyi/DALLE/DALLE_" + str(size) + "_JPG/"
                self.Pos_root = "/data/xiziyi/DALLE/ALASKA_" + str(size) + "_JPG/"
        elif type == "DreamStudio":
            if option is not None:
                self.Neg_root = os.path.join("/data/xiziyi/ds/DS_"+ str(size) +"_JPG_postpro/DS/", option) + '/'
                self.Pos_root = os.path.join("/data/xiziyi/ds/DS_"+ str(size) + "_JPG_postpro/ALASKA/", option) + '/'
            else:
                self.Neg_root = "/data/xiziyi/ds/DS_" + str(size) + "_JPG/"
                self.Pos_root = "/data/xiziyi/DALLE/ALASKA_"+ str(size) + "_JPG/"
        elif (type == "SPL2018" or type == 'DsTok') and option is not None:
            self.Neg_root = "/data/xiziyi/" + type + '/' + "split0/224_JPG_postpro/" + option + '/' + 'CG/'
            self.Pos_root = "/data/xiziyi/" + type + '/' + "split0/224_JPG_postpro/" + option + '/' + 'PG/'
            #print(self.Neg_root)
            #print(self.Pos_root)
        self.type = type
        self.image_list = self.get_dataset_info()
        self.augment = augment
        random.shuffle(self.image_list)
        self.tensor = transforms.ToTensor()
        if transforms:
            self.transforms = trans

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, item):
        img_path, label = self.image_list[item]
        image = Image.open(img_path).convert('RGB')
        #resize 256 or center crop
        '''
        if self.augment:
            method = random.randint(0, 7)
            #print(method)
            if method == 1:
                num = round(random.uniform(0.5, 2.5), 2)
                image = ImageEnhance.Color(image).enhance(num)
            elif method == 2:
                num = round(random.uniform(0.5, 2.5), 2)
                image = ImageEnhance.Brightness(image).enhance(num)
            elif method == 3:
                num = round(random.uniform(0.5, 2.5), 2)
                image = ImageEnhance.Contrast(image).enhance(num)
            elif method == 4:
                num = random.randint(0, 4)
                image = ImageEnhance.Sharpness(image).enhance(num)
            elif method == 5:
                img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                angle = random.randint(-180, 180)
                image = Image.fromarray(cv2.cvtColor(imutils.rotate(img, angle), cv2.COLOR_BGR2RGB))
            elif method == 6:
                img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                num = random.randint(0, 1)
                if num == 0:
                    img = cv2.GaussianBlur(img, (5, 5), 0)
                elif num == 1:
                    img = cv2.GaussianBlur(img, (3, 3), 0)
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            elif method == 7:
                img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                num = random.randint(0, 1)
                if num == 0:
                    img = cv2.blur(img, (5, 5), 0)
                elif num == 1:
                    img = cv2.blur(img, (3, 3), 0)
                image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if self.type != "DALLE" or self.type != "DreamStudio":
            image = self.tensor(image)
        
        if image.width != 256 or image.height != 256:
            padding_w = 0
            padding_h = 0
            if image.width < 256:
                padding_w = ceil((256 - image.width) / 2)
            if image.height < 256:
                padding_h = ceil((256 - image.height) / 2)
            if padding_h != 0 or padding_w != 0:
                image = ImageOps.expand(image, (padding_w, padding_h, padding_w, padding_h))
            #image = cv2.copyMakeBorder(image, padding_h, padding_h, padding_w, padding_w,cv2.BORDER_REFLECT)
        startx = (image.height - 256) // 2
        starty = (image.width - 256) // 2
        image = image.crop((starty, startx, starty+256,startx+256))
        '''
        image = self.transforms(image)
        if label == "0":
            label = 0
        else:
            label = 1
        #torch.as_tensor(image)
        label = torch.as_tensor(label)
        return image, label

    def get_dataset_info(self):
        image_list = []
        with open(self.root,'r') as file:
            lines = file.read().splitlines()
        if (self.type == 'SPL2018' or self.type == 'DsTok') and (self.option is None):
            for line in lines:
                image_path = line[:-2]
                label = line[-1]
                image_list.append([image_path, label])
        else:
            #if self.type == 'DALLE' or self.type == 'DreamStudio':
            for line in lines:
                name = os.path.basename(line[:-2])
                #print(name)
                #if '.JPG' in name:
                #    name = name.replace('.JPG', '.jpg')
                # image_path.replace("openai_90","openai_75to85")
                label = line[-1]
                # 0: dalle
                if label == '0':
                    image_path = self.Neg_root + name
                # 1: alaska
                else:
                    image_path = self.Pos_root + name
                image_list.append([image_path, label])
        return image_list