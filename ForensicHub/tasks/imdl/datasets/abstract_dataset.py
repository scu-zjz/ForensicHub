import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader

from .utils import pil_loader, denormalize

from .utils import EdgeMaskGenerator
from ..transforms import get_albu_transforms

from ForensicHub.core.base_dataset import BaseDataset


class AbstractDataset(BaseDataset):
    def _init_dataset_path(self):
        tp_path = None  # Tampered image
        gt_path = None  # Ground truth

        raise NotImplementedError  # abstract dataset!

        return tp_path, gt_path  # returns should be look like this

    def __init__(self, path,
                 is_padding=False,
                 is_resizing=False,
                 image_size=1024,
                 # output_size=(1024, 1024),
                 common_transform=None,
                 edge_mask_width=None,
                 img_loader=pil_loader,
                 post_funcs=None,
                 post_transform=None,
                 ) -> None:
        super().__init__(path, transform=common_transform, img_loader=img_loader, post_funcs=post_funcs,
                         post_transform=post_transform)

        output_size = (image_size, image_size)

        if is_padding == True and is_resizing == True:
            raise AttributeError("is_padding and is_resizing can not be True at the same time")
        if is_padding == False and is_resizing == False:
            raise AttributeError("is_padding and is_resizing can not be False at the same time")

        # Padding or Resizing
        self.post_transform = post_transform
        if is_padding == True:
            self.post_transform = get_albu_transforms(type_="pad", output_size=output_size)
        if is_resizing == True:
            self.post_transform = get_albu_transforms(type_="resize", output_size=output_size)
        self.is_padding = is_padding
        self.is_resizing = is_resizing

        self.output_size = output_size

        # Common augmentations for augmentation
        self.common_transform = common_transform
        # Edge mask generator
        self.edge_mask_generator = None if edge_mask_width is None else EdgeMaskGenerator(edge_mask_width)

        self.img_loader = img_loader
        self.post_funcs = post_funcs

    def __getitem__(self, index):

        data_dict = dict()

        tp_path = self.tp_path[index]
        gt_path = self.gt_path[index]

        # pil_loader or jpeg_loader
        tp_img = self.img_loader(tp_path)
        # shape, here is PIL Image
        tp_shape = tp_img.size

        # if "negative" then gt is a image with all 0
        if gt_path != "Negative":
            gt_img = self.img_loader(gt_path)
            gt_shape = gt_img.size
            label = 1
        else:
            temp = np.array(tp_img)
            gt_img = np.zeros((temp.shape[0], temp.shape[1], 3))
            gt_shape = (temp.shape[1], temp.shape[0])
            label = 0

        assert tp_shape == gt_shape, "tp and gt image shape must be the same, but got shape {} and {} for image '{}' and '{}'. Please check it!".format(
            tp_shape, gt_shape, tp_path, gt_path)

        tp_img = np.array(tp_img)  # H W C
        gt_img = np.array(gt_img)  # H W C

        # Do augmentations
        if self.common_transform != None:
            res_dict = self.common_transform(image=tp_img, mask=gt_img)
            tp_img = res_dict['image']
            gt_img = res_dict['mask']
            # copy_move may cause the label change, so we need to update the label
            if np.all(gt_img == 0):
                label = 0
            else:
                label = 1

        # redefine the shape, here is np.array
        tp_shape = tp_img.shape[0:2]  # H, W, 3 remove the last 3

        gt_img = (np.mean(gt_img, axis=2,
                          keepdims=True) > 127.5) * 1.0  # fuse the 3 channels to 1 channel, and make it binary(0 or 1)
        gt_img = gt_img.transpose(2, 0, 1)[0]  # H W C -> C H W -> H W
        masks_list = [gt_img]

        # if need to generate broaden edge mask
        if self.edge_mask_generator != None:
            gt_img_edge = self.edge_mask_generator(gt_img)[0][0]  # B C H W -> H W
            masks_list.append(gt_img_edge)  # albumentation interface
        else:
            pass

        # Do post-transform (paddings or resizing)
        res_dict = self.post_transform(image=tp_img, masks=masks_list)

        tp_img = res_dict['image']
        gt_img = res_dict['masks'][0].unsqueeze(0)  # H W -> 1 H W

        if self.edge_mask_generator != None:
            gt_img_edge = res_dict['masks'][1].unsqueeze(0)  # H W -> 1 H W

            # =========output=====================
            data_dict['edge_mask'] = gt_img_edge
            # ====================================
        # name of the image (mainly for testing)
        basename = os.path.basename(tp_path)

        # =========output=====================
        data_dict['image'] = tp_img
        data_dict['mask'] = gt_img
        data_dict['label'] = label
        data_dict['origin_shape'] = torch.tensor(
            tp_shape)  # (H, W) will become 3D matrix after data loader, 0th dim is batch_index
        # if resized
        if self.is_resizing:
            tp_shape = self.output_size

        # if (256, 384), then the image is a horizontal rectangle
        data_dict['shape'] = torch.tensor(
            tp_shape)  # (H, W) will become 3D matrix after data loader, 0th dim is batch_index
        data_dict['name'] = basename

        # if padding, need to return a shape_mask
        if self.is_padding:
            shape_mask = torch.zeros_like(gt_img)
            shape_mask[:, :tp_shape[0], :tp_shape[1]] = 1
            data_dict['shape_mask'] = shape_mask
        # ====================================
        # Post processing with callback functions on data_dict
        if self.post_funcs == None:
            pass  # Do nothing
        elif isinstance(self.post_funcs, list):
            # if it's a list, call each callback function in the list
            for func in self.post_funcs:
                if callable(func):
                    func(data_dict)
                else:
                    raise NotImplementedError(f"Element {func} in list is not callable")
        elif callable(self.post_funcs):
            # if it's a single callback function, call it directly
            self.post_funcs(data_dict)
        else:
            # other types raise NotImplementedError
            raise NotImplementedError(f"Unsupported type: {type(self.post_funcs)}")
        # ====================================

        return data_dict

    def __len__(self):
        return len(self.tp_path)
