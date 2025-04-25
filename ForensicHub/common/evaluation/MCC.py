import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import os
from IMDLBenCo.training_scripts.utils import misc
from IMDLBenCo.evaluation.abstract_class import AbstractEvaluator
from ForensicHub.registry import register_evaluator


@register_evaluator("ImageMCC")
class ImageMCC(AbstractEvaluator):
    def __init__(self, threshold=0.5) -> None:
        super().__init__()
        self.name = "image-level MCC"
        self.desc = "image-level Matthews Correlation Coefficient"
        self.threshold = threshold
        self.predict = []
        self.label = []
        self.remain_predict = []
        self.remain_label = []
        self.world_size = misc.get_world_size()

    def batch_update(self, predict_label, label, *args, **kwargs):
        self.predict.append(predict_label)
        self.label.append(label)

    def remain_update(self, predict_label, label, *args, **kwargs):
        self.remain_predict.append(predict_label)
        self.remain_label.append(label)

    def epoch_update(self):
        predict = torch.cat(self.predict, dim=0)
        label = torch.cat(self.label, dim=0)
        gather_predict = [torch.zeros_like(predict) for _ in range(self.world_size)]
        gather_label = [torch.zeros_like(label) for _ in range(self.world_size)]
        dist.all_gather(gather_predict, predict)
        dist.all_gather(gather_label, label)
        gather_predict = torch.cat(gather_predict, dim=0)
        gather_label = torch.cat(gather_label, dim=0)
        if self.remain_predict:
            gather_predict = torch.cat([gather_predict, torch.cat(self.remain_predict, dim=0)], dim=0)
            gather_label = torch.cat([gather_label, torch.cat(self.remain_label, dim=0)], dim=0)

        pred = (gather_predict > self.threshold).float()
        TP = torch.sum(pred * gather_label)
        TN = torch.sum((1 - pred) * (1 - gather_label))
        FP = torch.sum(pred * (1 - gather_label))
        FN = torch.sum((1 - pred) * gather_label)

        numerator = TP * TN - FP * FN
        denominator = torch.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) + 1e-9
        mcc = numerator / denominator
        return mcc

    def recovery(self):
        self.predict = []
        self.label = []
        self.remain_predict = []
        self.remain_label = []
