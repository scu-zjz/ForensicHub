from collections.abc import Callable
from collections import abc
from typing import Dict, List, Optional, Type, Union, Any
import inspect

import difflib
from rich.console import Console
from rich.table import Table

from ForensicHub.core.base_dataset import BaseDataset
from ForensicHub.core.base_model import BaseModel
from ForensicHub.core.base_transform import BaseTransform


class Registry:
    """A registry to map strings to classes or functions.

    Registered object could be built from registry. Meanwhile, registered
    functions could be called from registry.
    """

    def __init__(self, name: str):
        self._name = name
        self._module_dict: Dict[str, Type] = dict()

    def __len__(self):
        return len(self._module_dict)

    def __contains__(self, key):
        return self.get(key) is not None

    def __repr__(self):
        table = Table(title=f'Registry of {self._name}')
        table.add_column('Names', justify='left', style='cyan')
        table.add_column('Objects', justify='left', style='green')

        for name, obj in sorted(self._module_dict.items()):
            table.add_row(name, str(obj))

        console = Console()
        with console.capture() as capture:
            console.print(table, end='')

        return capture.get()

    @property
    def name(self):
        return self._name

    def has(self, name: str) -> bool:
        """Check if a name is in the registry."""
        return name in self._module_dict

    @property
    def module_dict(self):
        return self._module_dict

    def _suggest_correction(self, input_string: str) -> Optional[str]:
        """Suggest the most similar string from the registered modules."""
        suggestions = difflib.get_close_matches(input_string, self._module_dict.keys(), n=1, cutoff=0.6)
        if suggestions:
            return suggestions[0]
        return None

    def get(self, name):
        if name in self._module_dict:
            return self._module_dict[name]
        suggestion = self._suggest_correction(name)
        if suggestion:
            raise KeyError(f'"{name}" is not registered in {self.name}. Did you mean "{suggestion}"?')
        else:
            raise KeyError(f'"{name}" is not registered in {self.name} and no similar names were found.')

    def _register_module(self,
                         module: Type,
                         module_name: Optional[Union[str, List[str]]] = None,
                         force: bool = False) -> None:
        """Register a module.

        Args:
            module (type): Module to be registered. Typically a class or a
                function, but generally all ``Callable`` are acceptable.
            module_name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
                Defaults to None.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
        """
        if not callable(module):
            raise TypeError(f'module must be Callable, but got {type(module)}')

        if module_name is None:
            module_name = module.__name__
        if isinstance(module_name, str):
            module_name = [module_name]
        for name in module_name:
            if not force and name in self._module_dict:
                existed_module = self.module_dict[name]
                raise KeyError(f'{name} is already registered in {self.name} '
                               f'at {existed_module.__module__}')
            self._module_dict[name] = module

    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = False,
            module: Optional[Type] = None) -> Union[type, Callable]:
        """Register a module.

        A record will be added to ``self._module_dict``, whose key is the class
        name or the specified name, and value is the class itself.
        It can be used as a decorator or a normal function.

        Args:
            name (str or list of str, optional): The module name to be
                registered. If not specified, the class name will be used.
            force (bool): Whether to override an existing class with the same
                name. Defaults to False.
            module (type, optional): Module class or function to be registered.
                Defaults to None.
        """
        if not isinstance(force, bool):
            raise TypeError(f'force must be a boolean, but got {type(force)}')

        if not (name is None or isinstance(name, str) or is_seq_of(name, str)):
            raise TypeError(
                'name must be None, an instance of str, or a sequence of str, '
                f'but got {type(name)}')

        if module is not None:
            self._register_module(module=module, module_name=name, force=force)
            return module

        def _register(module):
            self._register_module(module=module, module_name=name, force=force)
            return module

        return _register

    def build(self, name: dict, *args, **kwargs) -> Any:
        """Build an instance.

        Build an instance by calling :attr:`build_func`.

        Args:
            cfg (dict): Config dict needs to be built.

        Returns:
            Any: The constructed object.
        """
        return self.get(name)(*args, **kwargs)


# 创建注册器实例
MODELS = Registry(name='MODELS')
POSTFUNCS = Registry(name='POSTFUNCS')
DATASETS = Registry(name='DATASETS')
TRANSFORMS = Registry(name='TRANSFORMS')
EVALUATORS = Registry(name='EVALUATORS')

# 注册基础抽象类
MODELS.register_module(module=BaseModel, name='BaseModel')
DATASETS.register_module(module=BaseDataset, name='BaseDataset')
TRANSFORMS.register_module(module=BaseTransform, name='BaseTransform')


# 添加便捷的装饰器函数
def register_model(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """Register a model."""
    return MODELS.register_module(name=name, force=force)


def register_dataset(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """Register a dataset."""
    return DATASETS.register_module(name=name, force=force)


def register_transform(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """Register a transform."""
    return TRANSFORMS.register_module(name=name, force=force)


def register_postfunc(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """Register a post-processing function."""
    return POSTFUNCS.register_module(name=name, force=force)


def register_evaluator(name: Optional[Union[str, List[str]]] = None, force: bool = False):
    """Register a post-processing function."""
    return EVALUATORS.register_module(name=name, force=force)


def build_from_registry(registry, config_args):
    # 从字典中获取 class 名称
    name = config_args["name"]  # 直接从字典中访问
    cls = registry.get(name)
    if cls is None:
        raise ValueError(f"Class '{name}' not found in registry.")

    # 获取 config 字典中的参数
    if "init_config" in config_args:
        config = config_args.get("init_config", {})  # 从字典中获取 config 部分
    else:
        config = {}

    # 处理额外的参数：比如如果参数是字符串 "true"，可以转化为布尔值
    for k, v in config.items():
        if isinstance(v, str):
            if v.lower() == "true":
                config[k] = True
            elif v.lower() == "false":
                config[k] = False

    print(f"[build_from_registry] Creating model '{name}' with args: {config}")
    return cls(**config)


def is_seq_of(seq: Any,
              expected_type: Union[Type, tuple],
              seq_type: Type = None) -> bool:
    """Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type or tuple): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Defaults to None.

    Returns:
        bool: Return True if ``seq`` is valid else False.

    Examples:
        >>> from mmengine.utils import is_seq_of
        >>> seq = ['a', 'b', 'c']
        >>> is_seq_of(seq, str)
        True
        >>> is_seq_of(seq, int)
        False
    """
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True
