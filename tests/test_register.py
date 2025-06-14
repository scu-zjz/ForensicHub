import sys

sys.path.append(".")
from ForensicHub.registry import DATASETS, MODELS, POSTFUNCS, TRANSFORMS, EVALUATORS

print(DATASETS._module_dict.keys())
print(MODELS._module_dict.keys())
print(POSTFUNCS._module_dict.keys())
print(TRANSFORMS._module_dict.keys())
print(EVALUATORS._module_dict.keys())
