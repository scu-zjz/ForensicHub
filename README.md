# ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization

**ForensicHub** is the go-to benchmark and modular codebase for all-domain fake image detection and localization,
covering deepfake detection (Deepfake), image manipulation detection and localization (IMDL), artificial
intelligence-generated image detection (AIGC), and document image manipulation localization (Doc). Whether you're
benchmarking
forensic models or building your own cross-domain pipelines, **ForensicHub** offers a flexible, configuration-driven
architecture to streamline development, comparison, and analysis.

🕵️‍♂️ **ForensicHub provides four core modular components:**

### 🗂️ Datasets

Datasets handle the data loading process and are required to return fields that conform to the ForensicHub
specification.

### 🔧 Transforms

Transforms handle the data pre-processing and augmentation for different tasks.

### 🧠 Models

Models, through alignment with Datasets and unified output, allow for the inclusion of various
state-of-the-art image forensic models.

### 📊 Evaluators

Evaluators cover commonly used image- and pixel-level metrics for different tasks, and are implemented with GPU
acceleration to improve evaluation efficiency during training and testing.


![](./images/overview.png)


## Installation

---

我们提供两种方式使用ForensicHub，通过Python Package安装，或者通过Clone项目到本地。

### Python Package
With `pip` :
```
TBD
```
With `conda` :
```
TBD
```

### Clone
直接命令行输入：
```
git clone https://github.com/scu-zjz/ForensicHub.git
```

## Quick Start

---



[//]: # (## 开发用链接：)

[//]: # (- [文档Github仓库]&#40;https://github.com/scu-zjz/ForensicHub-doc&#41;)

[//]: # (- [文档的主页]&#40;https://scu-zjz.github.io/ForensicHub-doc/&#41;)

[//]: # (- [PyPi]&#40;https://pypi.org/project/forensichub/&#41;)

[//]: # (## Reference链接)

[//]: # (- [DeepfakeBench原生仓库]&#40;https://github.com/SCLBD/DeepfakeBench&#41;)

[//]: # (- [DeepfakeBench我们版本]&#40;https://github.com/scu-zjz/DeepfakeBench&#41;)

[//]: # (- [AIGCBench]&#40;https://github.com/Ekko-zn/AIGCDetectBenchmark?tab=readme-ov-file&#41;)

[//]: # ()

[//]: # (## Local install)

[//]: # (本地开发者安装（实时更新）)

[//]: # (需要先切换到clone下来的ForensicHub的路径下，然后执行如下命令)

[//]: # (```shell)

[//]: # (pip install -e .)

[//]: # (```)

[//]: # (目前pypi仅仅用于站坑，暂时不要从pypi直接安装。)

[//]: # ()

[//]: # (## One-line Training/testing)

[//]: # (```)

[//]: # (forhub train /mnt/data0/xiaochen/workspace/fornhub/ForensicHub/ForensicHub/statics/aigc/train_resnet.yaml)

[//]: # (```)

[//]: # ()

[//]: # (```)

[//]: # (forhub test /mnt/data0/xiaochen/workspace/fornhub/ForensicHub/ForensicHub/statics/aigc/test_resnet.yaml)

[//]: # (```)

[//]: # ()

[//]: # (## IMDLBenCo式的代码生成和Training)

[//]: # (找一个干净的工作路径，然后执行如下指令：)

[//]: # (```)

[//]: # (forhub init)

[//]: # (```)

[//]: # ()

[//]: # (这样就会在这个路径下生成所需的yaml和shell脚本，其中`run.sh`作为全局入口，这个模式鼓励任意修改代码。)

[//]: # ()

[//]: # (后续可能会添加`forhub init imdl` `forhub init aigc`这样的分支入口应对不同的情况。)

[//]: # ()

[//]: # (## Command Line)

[//]: # (查看版本)

[//]: # (```)

[//]: # (forhub -v )

[//]: # (```)

[//]: # ()

[//]: # (初始化（目前没有实现功能）：)

[//]: # (```)

[//]: # (forhub init)

[//]: # (```)

[//]: # ()

[//]: # ()

[//]: # (## TODO list)

[//]: # (- [ ] 完善文档)

[//]: # (- [ ] 确定所有CMD的功能和接口API形式)

[//]: # (- [ ] 缝合朱哥的Deepfake部分)

[//]: # (- [ ] 测试image-level的F1分数。)

[//]: # ()

[//]: # (## Repo Structure:)

[//]: # (- common # 通用组件)

[//]: # (- core # 核心组件，用户接入时需要继承base_dataset,base_model,base_transform三个组件并注册，用户需要自己保证使用的dataset输出与model的输入对应)

[//]: # (- tasks # 不同垂类任务的组件库)

[//]: # (- training_scripts # 训练和测试入口，以yaml格式配置管理，统一入口为run.sh)
