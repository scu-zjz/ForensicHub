from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import albumentations as albu

class BaseTransform(ABC):
    """Base class for all transforms.
    
    This class defines the basic interface for transforms in ForensicsHub.
    Subclasses should implement get_train_transform and get_test_transform methods
    to return albumentations Compose objects.
    """
    
    def __init__(self):
        pass
        
    @abstractmethod
    def get_train_transform(self) -> albu.Compose:
        """Get training transforms.
        
        Returns:
            albu.Compose: Albumentations Compose object containing training transforms.
        """
        raise NotImplementedError("Subclasses must implement get_train_transform method")
        
    @abstractmethod
    def get_test_transform(self) -> albu.Compose:
        """Get testing transforms.
        
        Returns:
            albu.Compose: Albumentations Compose object containing testing transforms.
        """
        raise NotImplementedError("Subclasses must implement get_test_transform method")
        
    def __call__(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Apply transform to data dictionary.
        
        Args:
            data_dict (Dict[str, Any]): Dictionary containing input data.
                Must contain at least:
                    - 'image': Input image tensor
                May contain:
                    - 'mask': Ground truth mask
                    - 'label': Ground truth label
                    - Other transform-specific inputs
                    
        Returns:
            Dict[str, Any]: Dictionary containing transformed data.
                Must contain at least:
                    - 'image': Transformed image tensor
                May contain:
                    - 'mask': Transformed mask
                    - 'label': Transformed label
                    - Other transform-specific outputs
        """
        # By default, use training transforms
        transform = self.get_train_transform()
        
        # Apply transform
        transformed = transform(**data_dict)
        
        return transformed
        
    def __str__(self) -> str:
        """Return a string representation of the transform."""
        return self.__class__.__name__
