from .datasets import *
from .models import *
from .transforms import *

from IMDLBenCo.model_zoo import __all__
from IMDLBenCo.model_zoo import *
from ForensicHub import MODELS

# 确保模型类注册到框架
for model_name in __all__:
    # 获取模型类
    model_class = globals().get(model_name)  # 从全局字典中获取模型类

    # 确保模型类存在且是一个类
    if isinstance(model_class, type):
        # 检查模型类是否来自 IMDLBenCo.models 模块
        if model_class.__module__.startswith('IMDLBenCo.model_zoo'):
            # 注册模型
            MODELS.register_module(model_name, force=True, module=model_class)
