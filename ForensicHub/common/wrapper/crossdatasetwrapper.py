import random
from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.registry import register_dataset, build_from_registry, DATASETS


@register_dataset("CrossDataset")
class CrossDataset(BaseDataset):
    def __init__(self, dataset_config=[], **kwargs):
        super().__init__(path='', **kwargs)
        self.datasets = []
        self.pic_nums = []
        self.dataset_names = []
        self.return_mask = True
        for config in dataset_config:
            config['init_config']['common_transform'] = self.common_transform
            config['init_config']['post_transform'] = self.post_transform
            config['init_config']['post_funcs'] = self.post_funcs
            dataset = build_from_registry(DATASETS, config)
            self.dataset_names.append(config['name'])
            self.datasets.append(dataset)
            self.pic_nums.append(config['pic_nums'])

        # 数据集的最小keys集合
        self.common_keys = self.get_common_keys(self.datasets)
        print(self.common_keys)

    def __len__(self):
        total_samples = sum(self.pic_nums)  # 每个数据集的 pic_num 加起来
        return total_samples

    def _init_dataset_path(self) -> None:
        pass

    def get_common_keys(self, datasets):
        """
        获取所有数据集样本中 key 的最小公共集合
        """
        try:
            # 初始化为第一个数据集的第一个样本的 key 集合
            common_keys = set(datasets[0][0].keys())
            for dataset in datasets[1:]:
                sample = dataset[0]
                common_keys &= set(sample.keys())
            return common_keys
        except Exception as e:
            return set()  # 如果出错，返回空集合

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
                origin_out_dict = {key: origin_out_dict[key] for key in self.common_keys if key in origin_out_dict}
                origin_out_dict['label'] = origin_out_dict['label'].long()
                return origin_out_dict

        raise IndexError("Index out of range")

    def __str__(self):
        # 打印 CrossDataset 信息
        info = f"<===CrossDataset with {len(self.datasets)} datasets: {str(self.dataset_names)}===>\n"
        for i, ds in enumerate(self.datasets):
            info += f"  └─ Dataset {i}: {len(ds):,} samples, random sample: {self.pic_nums[i]}\n"
        info += f"Total samples per epoch: {self.__len__():,}\n"
        info += f"<================================================>\n"
        return info
