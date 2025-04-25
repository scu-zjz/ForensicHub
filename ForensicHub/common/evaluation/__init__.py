import IMDLBenCo.evaluation
from IMDLBenCo.evaluation import *
from ForensicHub import EVALUATORS
from .AP import ImageAP
from .MCC import ImageMCC
from .TNR import ImageTNR
from .TPR import ImageTPR

# 确保 AbstractEvaluator 被导入，用于后续检查
from IMDLBenCo.evaluation.abstract_class import AbstractEvaluator

# 动态注册所有 AbstractEvaluator 的非抽象子类
for name in dir():
    obj = globals()[name]
    if isinstance(obj, type) and issubclass(obj, AbstractEvaluator):
        # 确保类来自 IMDLBenCo.evaluation 模块
        if obj.__module__.startswith('IMDLBenCo.evaluation'):
            EVALUATORS.register_module(name, force=True, module=obj)

__all__ = IMDLBenCo.evaluation.__all__
