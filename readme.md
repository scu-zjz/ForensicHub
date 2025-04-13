Structure:
- common # 通用组件
- core # 核心组件，用户接入时需要继承base_dataset,base_model,base_transform三个组件并注册，用户需要自己保证使用的dataset输出与model的输入对应
- tasks # 不同垂类任务的组件库
- training_scripts # 训练和测试入口，以yaml格式配置管理，统一入口为run.sh
