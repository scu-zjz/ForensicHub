import json
import os
import random
import torch
import numpy as np
from PIL import Image
from ForensicHub.core.cross_dataset import BaseCrossDataset
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset


@register_dataset("DocumentCrossDataset")
class DocumentCrossDataset(BaseDataset):
    def __init__(self, path, image_size=224, **kwargs):
        self.image_size = image_size
        super().__init__(path=path, **kwargs)

    def _init_dataset_path(self) -> None:
        """Read dataset directories and parse image paths, masks, and labels."""
        all_samples = []

        # 确保 path 是 list 类型（兼容 str）
        path_list = [self.path] if isinstance(self.path, str) else self.path

        for path_id, one_path in enumerate(path_list):
            if not os.path.exists(one_path):
                raise FileNotFoundError(f"Dataset folder not found at {one_path}")

            image_dir = os.path.join(one_path, 'images')
            mask_dir = os.path.join(one_path, 'masks')

            if not os.path.exists(image_dir) or not os.path.exists(mask_dir):
                raise FileNotFoundError(f"Missing 'images' or 'masks' folder in {one_path}")

            image_list = sorted(os.listdir(image_dir))
            mask_list = sorted(os.listdir(mask_dir))

            assert len(image_list) == len(mask_list), \
                f"Mismatch in number of images and masks in {one_path}"

            for i in range(len(image_list)):
                sample = {
                    'path': os.path.join(image_dir, image_list[i]),
                    'mask': os.path.join(mask_dir, mask_list[i]),
                    'label': None,
                    # 'index': i,
                    # 'source': one_path  # 可选：记录来源路径
                }
                all_samples.append(sample)

        self.samples = all_samples
        self.entry_path = self.path  # For __str__()

    def __len__(self):
        return len(self.samples)
    
    # def _get_random_samples(self) -> list:
    #     """每个epoch随机选择pic_num张数据。"""
    #     if self.pic_num >=0:
    #         return random.sample(self.samples, self.pic_num)
    #     else:
    #         self.samples

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["path"]
        mask_path = sample["mask"]
        label = sample["label"]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)
        
        mask = Image.open(mask_path).convert("L")  # 'L'模式表示灰度图
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        mask = np.array(mask)

        # Apply transforms
        if self.common_transform:
            out = self.common_transform(image=image, mask=mask)
            image = out['image']
            mask = out['mask']
        if self.post_transform:
            out = self.post_transform(image=image, mask=mask)
            image = out['image']
            mask = out['image']
        mask = torch.tensor(mask, dtype=torch.float)
        output = {
            "image": image,
            "mask": mask,
            "label": torch.tensor(1).long() if torch.sum(mask) > 0 else torch.tensor(0).long()
        }

        return output

if __name__ == '__main__':
    # 这是pixel level 的mask的训练集参数 全假图  测试集对应路径旁边的test

    dataset = DocumentCrossDataset(path=['/mnt/data1/public_datasets/Doc/DocTamperV1-TrainingSet_reset_to_realworld','/mnt/data1/public_datasets/Doc/cutted_datasets_fakes/OSTF_train','/mnt/data1/public_datasets/Doc/cutted_datasets_fakes/RealTextManipulation_train','/mnt/data1/public_datasets/Doc/cutted_datasets_fakes/T-SROIE_train','/mnt/data1/public_datasets/Doc/cutted_datasets_fakes/Tampered-IC13_train'])
    # 总共七个测试集 DocTamperV有三个测试集 剩下四个数据集 分别是有一个
    # Doctamper的三个测试集分别如下 /mnt/data1/public_datasets/Doc/DocTamperV1/DocTamperV1-TestingSet /mnt/data1/public_datasets/Doc/DocTamperV1/DocTamperV1-FCD /mnt/data1/public_datasets/Doc/DocTamperV1/DocTamperV1-SCD
    # 剩下的为
    # /mnt/data1/public_datasets/Doc/cutted_datasets_fakes/OSTF_test
    # /mnt/data1/public_datasets/Doc/cutted_datasets_fakes/RealTextManipulation_test
    # /mnt/data1/public_datasets/Doc/cutted_datasets_fakes/T-SROIE_test
    # /mnt/data1/public_datasets/Doc/cutted_datasets_fakes/Tampered-IC13_test


    # 如果做image-level
    # 那就是把cutted_datasets_fakes换成cutted_datasets_alls  也就是全集 但是Doctamper仍然是只有假图 也训练带上
    # path=['/mnt/data1/public_datasets/Doc/DocTamperV1-TrainingSet_reset_to_realworld','/mnt/data1/public_datasets/Doc/cutted_datasets_alls/OSTF_train','/mnt/data1/public_datasets/Doc/cutted_datasets_alls/RealTextManipulation_train','/mnt/data1/public_datasets/Doc/cutted_datasets_alls/T-SROIE_train','/mnt/data1/public_datasets/Doc/cutted_datasets_alls/Tampered-IC13_train']
    # 测试集如下
    # Doctamper的三个测试集分别如下 /mnt/data1/public_datasets/Doc/DocTamperV1/DocTamperV1-TestingSet /mnt/data1/public_datasets/Doc/DocTamperV1/DocTamperV1-FCD /mnt/data1/public_datasets/Doc/DocTamperV1/DocTamperV1-SCD
    # 剩下的为
    # /mnt/data1/public_datasets/Doc/cutted_datasets_alls/OSTF_test
    # /mnt/data1/public_datasets/Doc/cutted_datasets_alls/RealTextManipulation_test
    # /mnt/data1/public_datasets/Doc/cutted_datasets_alls/T-SROIE_test
    # /mnt/data1/public_datasets/Doc/cutted_datasets_alls/Tampered-IC13_test
    dataset.__getitem__(0)
    import pdb;pdb.set_trace()

    