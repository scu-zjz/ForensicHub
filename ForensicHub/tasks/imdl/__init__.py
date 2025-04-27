from .datasets import *
from .models import *
from .transforms import *
import types

from IMDLBenCo.model_zoo import __all__
from IMDLBenCo.model_zoo import *
from ForensicHub import MODELS, POSTFUNCS

# 确保模型类注册到框架
for model_name in __all__:
    # 获取对象
    obj = globals().get(model_name)

    if obj is None:
        continue  # 没找到，跳过

    if isinstance(obj, type):
        # 是类，注册到 MODELS
        if obj.__module__.startswith('IMDLBenCo.model_zoo'):
            MODELS.register_module(model_name, force=True, module=obj)
    elif isinstance(obj, types.FunctionType):
        # 是函数，注册到 POSTFUNCS
        POSTFUNCS.register_module(model_name, force=True, module=obj)
    else:
        # 其他情况，可以根据需要提示或跳过
        pass
