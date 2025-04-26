from .version import __version__, version_info
from .registry import Registry, MODELS, DATASETS, POSTFUNCS, TRANSFORMS, EVALUATORS
from .common.backbones import *
from .common.evaluation import *

from .tasks import *
from .core import *
# make sure wrapper to be registered after models
from .common.wrapper import *

__all__ = ['__version__', 'version_info', 'MODELS', "DATASETS", "POSTFUNCS", "TRANSFORMS", "EVALUATORS"]
