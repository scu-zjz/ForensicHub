from IMDLBenCo.registry import MODELS
import torch.nn as nn
import torch
import torch.nn.functional as F

class DeepfakeOutputWrapper(nn.Module):
    def __init__(self, base_model) -> None:
        """
        The parameters of the `__init__` function will be automatically converted into the parameters expected by the argparser in the training and testing scripts by the framework according to their annotated types and variable names. 
        
        In other words, you can directly pass in parameters with the same names and types from the `run.sh` script to initialize the model.
        """
        super().__init__()
        
        # Useless, just an example
        self.base_model = base_model
    def forward(self, image, label, landmark, mask, *args, **kwargs):
        # import pdb;pdb.set_trace()
        
        data_dict = {'image':image, 'label':label, 'landmark':landmark, 'mask':mask}
        predictions = self.base_model(data_dict)
        losses = self.base_model.get_losses(data_dict, predictions)
        pred_label = predictions['prob']
        # ----------Output interface--------------------------------------
        output_dict = {
            # loss for backward
            "backward_loss": losses['overall'],
            # predicted mask, will calculate for metrics automatically
            "pred_mask": torch.randn(image.shape).to(image.device),
            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": 
                # keys here can be customized by yourself.
                losses
            ,
            "visual_image": {
            }
            # -------------------------------------------------------------
        }
        return output_dict
    
class BencoOutputWrapper(nn.Module):
    def __init__(self, base_model, model_args) -> None:
        """
        The parameters of the `__init__` function will be automatically converted into the parameters expected by the argparser in the training and testing scripts by the framework according to their annotated types and variable names. 
        
        In other words, you can directly pass in parameters with the same names and types from the `run.sh` script to initialize the model.
        """
        super().__init__()
        
        # Useless, just an example
        self.base_model = base_model
        self.model_args = model_args
        self.head = nn.AdaptiveAvgPool2d(1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, image, label, mask, landmark, *args, **kwargs):     
        if self.model_args['name'] in ['IML_ViT', 'Mesorch']:
            features = self.base_model.forward_features(image)
            pred_label = self.head(features).view(-1)
            loss = self.loss_fn(pred_label, label.float())
            pred_label = F.sigmoid(pred_label)
        else:
            mask = torch.randint(0,1,(image.shape[0],1,image.shape[2],image.shape[3])).long().to(image.device)
            edge_mask = torch.randint(0,1,(image.shape[0],1,image.shape[2],image.shape[3])).long().to(image.device)
            outputs = self.base_model(image=image, mask=mask, edge_mask=edge_mask,label=label, *args, **kwargs)
            pred_label = outputs['pred_label']
            loss = F.binary_cross_entropy(pred_label, label.float())
        # ----------Output interface--------------------------------------
        output_dict = {
            # loss for backward
            "backward_loss": loss,

            # predicted binaray label, will calculate for metrics automatically
            "pred_label": pred_label,

            # ----values below is for visualization----
            # automatically visualize with the key-value pairs
            "visual_loss": {
                'pred_loss':loss
            }
                # keys here can be customized by yourself.
            ,
            "visual_image": {
            }
            # -------------------------------------------------------------
        }
        return output_dict


def json_wrapper(config):
    return config