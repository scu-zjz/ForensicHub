import random
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset, build_from_registry, DATASETS


@register_dataset("CrossDataset")
class CrossDataset(BaseDataset):
    def __init__(self, dataset_config=[], **kwargs):
        self.datasets = []
        self.pic_nums = []
        self.dataset_names = []
        self.return_mask = True
        for config in dataset_config:
            config['common_transform'] = self.common_transform
            config['post_transform'] = self.post_transform
            config['post_funcs'] = self.post_funcs
            dataset = build_from_registry(DATASETS, config)
            self.dataset_names.append(config['name'])
            self.datasets.append(dataset)
            self.pic_nums.append(dataset.pic_num)
            # if any dataset doesn't return mask, CrossDataset will not return mask
            self.return_mask = self.return_mask and self._check_has_mask(dataset)
        super().__init__(path='', **kwargs)

    def __len__(self):
        total_samples = sum(self.pic_nums)  # 每个数据集的 pic_num 加起来
        return total_samples

    def _init_dataset_path(self) -> None:
        pass

    def _check_has_mask(self, dataset):
        # 校验数据集是否包含 mask
        try:
            # 尝试访问一个样本，如果样本中有 'mask'，返回 True，否则返回 False
            sample = dataset[0]
            if 'mask' in sample:
                return True
        except Exception as e:
            pass  # 如果无法访问，或者没有 'mask'，则返回 False
        return False

    def __getitem__(self, index):
        # 根据 index 确定应该从哪个数据集抽取图像
        # 累积的 pic_num 可以帮助我们确定当前从哪个数据集抽取
        cumulative_samples = 0

        for i, pic_num in enumerate(self.pic_nums):
            cumulative_samples += pic_num
            if index < cumulative_samples:  # 如果 index 在这个数据集的范围内
                selected_dataset = self.datasets[i]
                # 在当前数据集中随机选择一个样本
                selected_item = random.randint(0, len(selected_dataset) - 1)
                origin_out_dict = selected_dataset[selected_item]
                out_dict = {
                    'image': origin_out_dict['image'],
                    'label': origin_out_dict['label'],
                }
                if self.return_mask and origin_out_dict.get('mask') is not None:
                    out_dict['mask'] = origin_out_dict['mask']
                return out_dict

        raise IndexError("Index out of range")

    def __str__(self):
        # 打印 CrossDataset 信息
        info = f"<===CrossDataset with {len(self.datasets)} datasets: {str(self.dataset_names)}===>\n"

        info += f"Total samples per epoch: {self.__len__():,}\n"
        info += f"<================================================>\n"
        return info
