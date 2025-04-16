import yaml
from argparse import Namespace


def try_parse_value(v):
    # 尝试把字符串解析成 float，如果失败就返回原值
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    elif isinstance(v, dict):
        return {k: try_parse_value(val) for k, val in v.items()}
    elif isinstance(v, list):
        return [try_parse_value(item) for item in v]
    else:
        return v


def load_yaml_config(path):
    """加载 YAML 配置文件为 Python 字典，并自动将 '1e-4' 这种字符串转为 float"""
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    return try_parse_value(raw)


def dict_to_namespace(d):
    """递归地将 dict 转换成 argparse.Namespace"""
    if isinstance(d, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(x) for x in d]
    else:
        return d


def load_yaml_as_namespace(path):
    """一步到位：从 YAML 路径加载为 Namespace 对象"""
    config_dict = load_yaml_config(path)
    return dict_to_namespace(config_dict)


def split_config(config):
    """将配置字典划分为多个组件参数"""
    model_args = config.pop("model", {})
    train_dataset_args = config.pop("train_dataset", {})
    test_dataset_args = config.pop("test_dataset", [])
    transform_args = config.pop("transform", {})
    args = config  # 剩下的就是全局 args

    if "init_config" not in model_args:
        model_args["init_config"] = {}
    for x in test_dataset_args:
        if "init_config" not in x:
            x["init_config"] = {}
    if "init_config" not in train_dataset_args:
        train_dataset_args["init_config"] = {}
    if "init_config" not in transform_args:
        transform_args["init_config"] = {}

    return (
        dict_to_namespace(args),
        model_args,
        train_dataset_args,
        test_dataset_args,
        transform_args
    )

    # return (
    #     dict_to_namespace(args),
    #     dict_to_namespace(model_args),
    #     dict_to_namespace(train_dataset_args),
    #     [dict_to_namespace(td) for td in test_dataset_args],
    #     dict_to_namespace(transform_args),
    # )


def add_attr(ns, **kwargs):
    for k, v in kwargs.items():
        setattr(ns, k, v)
    return ns
