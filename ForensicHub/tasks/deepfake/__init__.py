from .datasets import *
from .models import *
from .transforms import *


from .datasets import *
from .models import *
from .transforms import *
import types

from DeepfakeBench.training import detectors
from DeepfakeBench.training.detectors import __all__
from ForensicHub import MODELS, POSTFUNCS
from .wrapper.wrappers import Deepfake2ForensicWrapper

for model_name in __all__:
    if not hasattr(detectors, model_name):
        continue

    obj = getattr(detectors, model_name)
    if not isinstance(obj, type):
        continue
    if not obj.__module__.startswith('DeepfakeBench.training.detectors'):
        continue

    def make_wrapped_class(base_cls, name):
        class WrappedModel(Deepfake2ForensicWrapper):
            def __init__(self, *args, **kwargs):
                super().__init__(base_cls, *args, **kwargs)
        WrappedModel.__name__ = f"Wrapped{name}"
        WrappedModel.__qualname__ = WrappedModel.__name__
        return WrappedModel

    wrapped_cls = make_wrapped_class(obj, model_name)
    MODELS.register_module(name=model_name, module=wrapped_cls, force=True)

    globals()[model_name] = wrapped_cls