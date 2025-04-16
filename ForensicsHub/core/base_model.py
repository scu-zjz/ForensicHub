import torch
import torch.nn as nn
from typing import Dict, Any, Optional


class BaseModel(nn.Module):
    """Base class for all models in ForensicsHub.
    
    This class defines the basic interface and common functionality
    for all models. The model should take at least an image as input
    and return a dictionary containing the necessary outputs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, **kwargs) -> Dict[str, Any]:
        """Forward pass of the model.
        
        Args:
            **kwargs (Dict[str, Any]): Dictionary containing input data.
                May contain (Consistent with the input from the dataset):
                    - 'image': Input image tensor
                    - 'mask': Ground truth mask
                    - 'label': Ground truth label
                    - Other model-specific inputs
                    
        Returns:
            Dict[str, Any]: Dictionary containing model outputs.
                Must contain at least:
                    - 'backward_loss': Backward loss value
                May contain:
                    - 'pred_mask': Predicted mask
                    - 'pred_label': Predicted label
                Visualize (Optional):
                    - 'visual_loss': Automatically visualize with the key-value pairs 
                    eg. {
                        "predict_loss": predict_loss,
                        "edge_loss": edge_loss,
                        "combined_loss": combined_loss
                    }
                    - 'visual_image': Automatically visualize with the key-value pairs
                    eg. {
                        "mask": mask,
                        "pred_mask": pred_mask,
                        "pred_label": pred_label
                    }
                    - Other model-specific outputs

                return eg.
                {
                    "backward_loss": combined_loss,

                    # optional below
                    "pred_mask": mask_pred,
                    "visual_loss": {
                        "combined_loss": combined_loss
                    },
                    "visual_image": {
                        "pred_mask": mask_pred,
                    }
                }
        """
        raise NotImplementedError("Subclasses must implement forward method")

    def get_prediction(self, data_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Get model prediction without computing loss.
        
        This is typically used during inference.
        
        Args:
            data_dict (Dict[str, Any]): Dictionary containing input data.
                Must contain at least:
                    - 'image': Input image tensor
                    
        Returns:
            Dict[str, Any]: Dictionary containing model outputs.
                Must contain at least:
                    - 'pred': Model prediction
        """
        with torch.no_grad():
            return self.forward(**data_dict)

    def compute_loss(self, data_dict: Dict[str, Any], output_dict: Dict[str, Any]) -> torch.Tensor:
        """Compute loss for training.
        
        Args:
            data_dict (Dict[str, Any]): Dictionary containing input data.
            output_dict (Dict[str, Any]): Dictionary containing model outputs.
                
        Returns:
            torch.Tensor: Loss value
        """
        raise NotImplementedError("Subclasses must implement compute_loss method")

    def get_metrics(self, data_dict: Dict[str, Any], output_dict: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics for evaluation.
        
        Args:
            data_dict (Dict[str, Any]): Dictionary containing input data.
            output_dict (Dict[str, Any]): Dictionary containing model outputs.
                
        Returns:
            Dict[str, float]: Dictionary containing metric names and values
        """
        raise NotImplementedError("Subclasses must implement get_metrics method")
