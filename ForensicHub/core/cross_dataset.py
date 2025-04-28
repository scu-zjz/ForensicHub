from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
from torch.utils.data import Dataset
from .base_dataset import BaseDataset


class BaseCrossDataset(BaseDataset):
    """Base cross class for all dataset to be used in the Cross Protocol.
    It inherits from torch.utils.data.Dataset and adds forensic-specific functionality.
    """

    def __init__(self,
                 path: str,
                 pic_num: int,
                 **kwargs):
        """Initialize the dataset.
        
        Args:
            path (str): Path to the dataset.
            pic_num (int): Numbers of pictures to be sampled from dataset in an epoch.
            **kwargs: Additional arguments specific to each dataset.
        """
        super().__init__(path=path)
        self.pic_num = pic_num

        # Initialize dataset paths
        self._init_dataset_path()

    @abstractmethod
    def _init_dataset_path(self) -> None:
        """Initialize dataset paths.
        
        This method should be implemented by each dataset class to set up
        the paths for images and labels/masks.
        """
        pass

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx (int): Index of the sample to get.
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing sample data.
            The dictionary should at least contain:
            - 'image': torch.Tensor, the input image
            May contain:
            - 'mask': torch.Tensor, the target mask
            - 'label': torch.Tensor, the target label
            Attention!!! No other parameters are allowed because items from different dataset will concat in a tensor
        """
        pass

    def __str__(self) -> str:
        """Return string representation of the dataset."""
        cls_name = self.__class__.__name__
        cls_path = self.entry_path
        cls_len = len(self)
        return f"[{cls_name}] at {cls_path}, with length of {cls_len:,}"
