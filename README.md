# ForensicHub: All-in-one solution for fake image detection!
## 开发用链接：
- [文档Github仓库](https://github.com/scu-zjz/ForensicHub-doc)
- [文档的主页](https://scu-zjz.github.io/ForensicHub-doc/)
- [PyPi](https://pypi.org/project/forensichub/)
## Reference链接
- [DeepfakeBench原生仓库](https://github.com/SCLBD/DeepfakeBench)
- [DeepfakeBench我们版本](https://github.com/scu-zjz/DeepfakeBench)
- [AIGCBench](https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file)

## Local install
本地开发者安装（实时更新）
需要先切换到clone下来的ForensicHub的路径下，然后执行如下命令
```shell
pip install -e .
```
目前pypi仅仅用于站坑，暂时不要从pypi直接安装。

## One-line Training/testing
```
forhub train /mnt/data0/xiaochen/workspace/fornhub/ForensicHub/ForensicHub/statics/aigc/train_resnet.yaml
```

```
forhub test /mnt/data0/xiaochen/workspace/fornhub/ForensicHub/ForensicHub/statics/aigc/test_resnet.yaml
```

## IMDLBenCo式的代码生成和Training
找一个干净的工作路径，然后执行如下指令：
```
forhub init
```

这样就会在这个路径下生成所需的yaml和shell脚本，其中`run.sh`作为全局入口，这个模式鼓励任意修改代码。

后续可能会添加`forhub init imdl` `forhub init aigc`这样的分支入口应对不同的情况。

## Command Line
查看版本
```
forhub -v 
```

初始化（目前没有实现功能）：
```
forhub init
```


## TODO list
- [ ] 完善文档
- [ ] 确定所有CMD的功能和接口API形式
- [ ] 缝合朱哥的Deepfake部分
- [ ] 测试image-level的F1分数。

## Repo Structure:
- common # 通用组件
- core # 核心组件，用户接入时需要继承base_dataset,base_model,base_transform三个组件并注册，用户需要自己保证使用的dataset输出与model的输入对应
- tasks # 不同垂类任务的组件库
- training_scripts # 训练和测试入口，以yaml格式配置管理，统一入口为run.sh
