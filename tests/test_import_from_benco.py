from ForensicHub.common.evaluation import *
import numpy as np
import torch
# 测试是否能import并计算dummy指标
f1_evaluator = PixelF1()
# 提供dummy参数：
dummy_pred = torch.rand((2, 1, 384, 256))
dummy_gt = torch.rand((2, 1, 384, 256))
dummy_region = torch.ones((2, 1, 384, 256))
# 计算dummy指标
TP, TN, FP, FN = f1_evaluator.Cal_Confusion_Matrix(dummy_pred, dummy_gt, shape_mask=dummy_region)
print(TP, TN, FP, FN)

