import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.distributed as dist
import os
from IMDLBenCo.training_scripts.utils import misc
from IMDLBenCo.evaluation.abstract_class import AbstractEvaluator
from ForensicHub.registry import register_evaluator


def average_precision_gpu(y_true, y_score):
    """
    GPU-compatible AP calculation.
    y_true: (N,) float tensor (0 or 1)
    y_score: (N,) float tensor (predicted scores)
    """
    # Sort by score descending
    sorted_idx = torch.argsort(y_score, descending=True)
    y_true = y_true[sorted_idx]
    y_score = y_score[sorted_idx]

    # True positives cumulative
    tp_cumsum = torch.cumsum(y_true, dim=0)
    total_positives = torch.sum(y_true)

    # Precision at each threshold
    precision = tp_cumsum / (torch.arange(1, len(y_true) + 1, device=y_true.device))

    # AP = sum(precision * delta_recall)
    delta_recall = y_true / total_positives
    ap = torch.sum(precision * delta_recall)
    return ap


@register_evaluator("ImageAP")
class ImageAP(AbstractEvaluator):
    def __init__(self) -> None:
        super().__init__()
        self.name = "image-level AP"
        self.desc = "image-level Average Precision"
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
        ap = average_precision_gpu(gather_label, gather_predict)
        return ap

    def recovery(self):
        self.predict = []
        self.label = []
        self.remain_predict = []
        self.remain_label = []
