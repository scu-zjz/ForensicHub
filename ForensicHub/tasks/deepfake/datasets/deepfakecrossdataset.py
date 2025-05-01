import json
import os
import random
import torch
import numpy as np
from PIL import Image
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset


# @register_dataset("DeepfakeCrossDataset")
class DeepfakeCrossDataset(BaseDataset):
    def __init__(self, path, config_file, compression='c23', image_size=256, split_mode='train', **kwargs):
        """
        Args:
            path (str): 根目录（包含图像文件的根路径，如 FaceForensics++/original_sequences/...）
            config_file (str): 指向 JSON 配置文件路径
            compression (str): 压缩类型，如 'c23' 或 'c40'
            image_size (int): 图像大小
        """
        self.path = path
        self.config_file = config_file
        self.compression = compression
        self.image_size = image_size
        self.split_mode = split_mode.lower()
        super().__init__(path=path, **kwargs)


    def _init_dataset_path(self):
        """从配置文件中读取图像路径和标签，支持 train/test 分支，压缩率字段可选。"""
        with open(self.config_file, 'r') as f:
            dataset_info = json.load(f)

        samples = []

        for dataset_name, label_groups in dataset_info.items():
            for label_str, splits in label_groups.items():
                label = 0 if "real" in label_str.lower() else 1  # 标签0=真实，1=伪造

                if self.split_mode not in splits:
                    continue
                # 如果有压缩率字段就使用，没有就直接取该分支
                if isinstance(splits[self.split_mode], dict) and self.compression in splits[self.split_mode]:
                    video_group = splits[self.split_mode][self.compression]
                elif isinstance(splits[self.split_mode], dict):
                    video_group = splits[self.split_mode]
                else:
                    continue  # 无法匹配则跳过

                for video_id, video_info in video_group.items():
                    frame_paths = video_info["frames"]
                    for rel_path in frame_paths:
                        full_path = os.path.join(self.path, rel_path)
                        samples.append({
                            "path": full_path,
                            "mask": None,
                            "label": label
                        })

        self.samples = samples


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["path"]
        label = sample["label"]

        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        image = np.array(image)

        mask = None  # 此任务中 mask 可为空
        
        if self.common_transform:
            out = self.common_transform(image=image)
            image = out['image']
        if self.post_transform:
            out = self.post_transform(image=image)
            image = out['image']

        return {
            "image": image,
            "mask": mask,  # 可以忽略不使用
            "label": torch.tensor(label).long()
        }

if __name__ == '__main__':
    dataset = DeepfakeCrossDataset(path='/mnt/data1/public_datasets/Deepfake', config_file='/mnt/data1/public_datasets/Deepfake/UADFV.json',split_mode='test')

    dataset.__getitem__(0)
    
    import pdb;pdb.set_trace()
    # path不变  唯一要变的是config_file  c40那个数据集compression变成'c40' 前面是名字 以及split_mode变为train或者test
    # 训练:FaceForensics++ /mnt/data1/public_datasets/Deepfake/FaceForensics++.json compression='c23'
    # 测试: 1. FaceForensics++  /mnt/data1/public_datasets/Deepfake/FaceForensics++.json
    # dataset = DeepfakeCrossDataset(path='/mnt/data1/public_datasets/Deepfake', config_file='/mnt/data1/public_datasets/Deepfake/FaceForensics++.json', compression='c40')
    # 2. FaceForensics++_40  /mnt/data1/public_datasets/Deepfake/FaceForensics++.json compression='c40'
    # dataset = DeepfakeCrossDataset(path='/mnt/data1/public_datasets/Deepfake', config_file='/mnt/data1/public_datasets/Deepfake/FF-DF.json',split_mode='test') 
    # len(dataset) == 8952
    # 3. FF-DF /mnt/data1/public_datasets/Deepfake/FF-DF.json  test
    # 4. FF-F2F /mnt/data1/public_datasets/Deepfake/FF-F2F.json test
    # 5.FF-FS /mnt/data1/public_datasets/Deepfake/FF-FS.json test
    # 6. FF-NT  /mnt/data1/public_datasets/Deepfake/FF-NT.json test
    # 7. Celeb-DF-v1 /mnt/data1/public_datasets/Deepfake/Celeb-DF-v1.json test
    # 8.Celeb-DF-v2 /mnt/data1/public_datasets/Deepfake/Celeb-DF-v2.json test
    # 9. DFD /mnt/data1/public_datasets/Deepfake/DeepFakeDetection.json test
    # 10. DFDC /mnt/data1/public_datasets/Deepfake/DFDC.json test
    # 11. DFDCP /mnt/data1/public_datasets/Deepfake/DFDCP.json test
    # 12. FaceShifter /mnt/data1/public_datasets/Deepfake/FaceShifter.json test
    # 13. UADFV /mnt/data1/public_datasets/Deepfake/UADFV.json test
