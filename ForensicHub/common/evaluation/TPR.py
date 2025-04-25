import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import os
from IMDLBenCo.training_scripts.utils import misc
from IMDLBenCo.evaluation.abstract_class import AbstractEvaluator
from ForensicHub.registry import register_evaluator


@register_evaluator("ImageTPR")
class ImageTPR(AbstractEvaluator):
    def __init__(self, threshold=0.5) -> None:
        super().__init__()
        self.name = "image-level TPR"
        self.desc = "image-level TPR"
        self.threshold = threshold
        self.predict = []
        self.label = []
        self.remain_label = []
        self.remain_predict = []
        self.world_size = misc.get_world_size()
        self.local_rank = misc.get_rank()

    def batch_update(self, predict_label, label, *args, **kwargs):
        self._chekc_image_level_params(predict_label, label)
        self.predict.append(predict_label)
        self.label.append(label)
        return None

    def remain_update(self, predict_label, label, *args, **kwargs):
        self.remain_predict.append(predict_label)
        self.remain_label.append(label)
        return None

    def epoch_update(self):
        if len(self.predict) != 0:
            predict = torch.cat(self.predict, dim=0)
            label = torch.cat(self.label, dim=0)
            gather_predict_list = [torch.zeros_like(predict) for _ in range(self.world_size)]
            gather_label_list = [torch.zeros_like(label) for _ in range(self.world_size)]
            dist.all_gather(gather_predict_list, predict)
            dist.all_gather(gather_label_list, label)
            gather_predict = torch.cat(gather_predict_list, dim=0)
            gather_label = torch.cat(gather_label_list, dim=0)
            if len(self.remain_predict) != 0:
                self.remain_predict = torch.cat(self.remain_predict, dim=0)
                self.remain_label = torch.cat(self.remain_label, dim=0)
                gather_predict = torch.cat([gather_predict, self.remain_predict], dim=0)
                gather_label = torch.cat([gather_label, self.remain_label], dim=0)
        else:
            if len(self.remain_predict) == 0:
                raise RuntimeError(f"No data to calculate {self.name}, please check the input data.")
            gather_predict = torch.cat(self.remain_predict, dim=0)
            gather_label = torch.cat(self.remain_label, dim=0)
        # calculate TPR
        predict = (gather_predict > self.threshold) * 1.0
        TP = torch.sum(predict * gather_label)
        FN = torch.sum((1 - predict) * gather_label)
        recall = TP / (TP + FN + 1e-9)
        return recall

    def recovery(self):
        self.predict = []
        self.label = []
        self.remain_label = []
        self.remain_predict = []
        return None
