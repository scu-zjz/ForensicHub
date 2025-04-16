# ForensicHub: All-in-one solution for fake image detection!
## Local install
```shell
pip install forensichub
```
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
