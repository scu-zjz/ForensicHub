from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import os
import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """Base class for all forensic datasets in ForensicHub.
    
    This class defines the interface that all dataset classes must implement.
    It inherits from torch.utils.data.Dataset and adds forensic-specific functionality.
    """

    def __init__(self,
                 path: Union[str, List[str]],
                 common_transform: Optional[Any] = None,
                 post_transform: Optional[Any] = None,
                 img_loader: Any = None,
                 post_funcs: Any = None,
                 **kwargs):
        """Initialize the dataset.
        
        Args:
            path (str): Path to the dataset.
            transform (Optional[Any]): Transform to be applied to the input data.
            img_loader (Any): Function to load images.
            post_funcs (Optional[List[callable]]): List of post-processing functions.
            **kwargs: Additional arguments specific to each dataset.
        """
        super().__init__()
        self.path = path
        self.common_transform = common_transform
        self.post_transform = post_transform
        self.img_loader = img_loader
        self.post_funcs = post_funcs

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
            Optional fields:
            - 'mask': torch.Tensor, the target mask
            - 'label': torch.Tensor, the target label
            - 'origin_shape': torch.Tensor, the original image shape
            - 'edge_mask': torch.Tensor, the edge mask of the image
            - ......
        """
        pass

    def __str__(self) -> str:
        """Return string representation of the dataset."""
        cls_name = self.__class__.__name__
        cls_path = self.path
        cls_len = len(self)
        return f"[{cls_name}] at {cls_path}, with length of {cls_len:,}"
