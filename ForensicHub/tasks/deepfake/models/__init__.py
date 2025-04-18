from ForensicHub import MODELS
from DeepfakeBench.training.detectors import DETECTOR
from DeepfakeBench.training.detectors.base_detector import AbstractDetector


for name, obj in DETECTOR.data.items():
    if isinstance(obj, type) and issubclass(obj, AbstractDetector):
        MODELS.register_module(name, force=True, module=obj)


if __name__ == "__main__":
    print(MODELS)