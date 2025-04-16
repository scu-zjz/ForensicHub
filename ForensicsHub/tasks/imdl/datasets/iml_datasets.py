import json
import os

from .abstract_dataset import AbstractDataset
from ForensicsHub.registry import register_dataset


@register_dataset('ManiDataset')
class ManiDataset(AbstractDataset):
    """Dataset for loading images from a directory structure with Tp and Gt folders.
    
    The directory structure should be:
    path/
        Tp/
            image1.jpg
            image2.jpg
            ...
        Gt/
            mask1.jpg
            mask2.jpg
            ...
    """

    def _init_dataset_path(self):
        # 检查目录结构
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"dataset path {self.path} not found")

        tp_dir = os.path.join(self.path, "Tp")
        gt_dir = os.path.join(self.path, "Gt")

        if not os.path.exists(tp_dir):
            raise FileNotFoundError(f"test images directory {tp_dir} not found")
        if not os.path.exists(gt_dir):
            raise FileNotFoundError(f"ground truth directory {gt_dir} not found")

        # 获取所有图片路径
        self.tp_path = []
        self.gt_path = []

        for filename in os.listdir(tp_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                tp_path = os.path.join(tp_dir, filename)
                gt_path = os.path.join(gt_dir, filename)

                if not os.path.exists(gt_path):
                    raise FileNotFoundError(f"ground truth {gt_path} not found")

                self.tp_path.append(tp_path)
                self.gt_path.append(gt_path)

        return self.tp_path, self.gt_path


@register_dataset('JsonDataset')
class JsonDataset(AbstractDataset):
    """Dataset for loading images from a JSON file.
    
    The JSON file should be organized as:
    [
        ["./Tp/6.jpg", "./Gt/6.jpg"],
        ["./Tp/7.jpg", "./Gt/7.jpg"],
        ["./Tp/8.jpg", "Negative"],
        ...
    ]
    If the second element is "Negative", it means the image is a negative sample
    with no manipulation.
    """

    def _init_dataset_path(self):
        # 读取json文件
        with open(self.path, 'r') as f:
            data = json.load(f)

        # 检查json格式
        if not isinstance(data, list):
            raise ValueError("json file should be a list of pairs")

        # 初始化路径列表
        self.tp_path = []
        self.gt_path = []

        # 遍历json数据
        for pair in data:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ValueError("each element in json file should be a pair")

            tp_path = pair[0]
            gt_path = pair[1]

            # 检查文件是否存在
            if not os.path.exists(tp_path):
                raise FileNotFoundError(f"test image {tp_path} not found")

            if gt_path != "Negative" and not os.path.exists(gt_path):
                raise FileNotFoundError(f"ground truth {gt_path} not found")

            self.tp_path.append(tp_path)
            self.gt_path.append(gt_path)

        return self.tp_path, self.gt_path

    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return f"{self.__class__.__name__}({len(self)} samples)"
