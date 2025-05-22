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

## 📁 Project Structure Overview

```bash
ForensicHub/
├── common/                 # Common modules
│   ├── backbones/          # Backbones and feature extractors
│   ├── evalaution/         # Image- and pixel-level evaluators
│   ├── utils/              # Utilities
│   └── wrapper/            # Wrappers for dataset, model, etc.
├── core/                   # Core module providing abstract base classes
├── statics/                # YAML configuration files for training and testing
├── tasks/                  # Components for different sub-tasks
│   ├── aigc/           
│   ├── deepfake/             
│   ├── document/            
│   └── imdl/     
└── training_scripts        # Scripts for training and evaluation
```

## Installation

---

We provide two ways to use ForensicHub: install via Python package or clone the project locally.

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

Simply run the following command:

```
git clone https://github.com/scu-zjz/ForensicHub.git
```

## Quick Start

---

The Quick Start example is based on the local clone setup. ForensicHub is a modular and configuration-driven lightweight
framework. You only need to use the built-in or custom Dataset, Transform, and Model components, register them, and then
launch the pipeline using a YAML configuration file.

<details>
<summary>Training on the DiffusionForensics dataset using Resnet for AIGC</summary>

1. Dataset Preparation

Download the DiffusionForensics dataset from (https://github.com/ZhendongWang6/DIRE).
The experiment only uses the ImageNet portion. Format the data as JSON. ForensicHub does not restrict how the data is
loaded—just make sure the Dataset returns fields as defined in `\core\base_dataset.py`. This means users are free to
implement their own loading logic. In this case, we
use `/tasks/aigc/datasets/label_dataset.py`, which expects a JSON with entries like with label of 0 and 1 representing a
image of real and generated:

```
[
  {
    "path": "/mnt/data3/public_datasets/AIGC/DiffusionForensics/images/train/imagenet/real/n03982430/ILSVRC2012_val_00039791.JPEG",
    "label": 0
  },
  {
    "path": "/mnt/data3/public_datasets/AIGC/DiffusionForensics/images/train/imagenet/real/n03982430/ILSVRC2012_val_00022594.JPEG",
    "label": 0
  },
  ...
]
```

2. Component Preparation

In this example, the **Model** is ResNet50, which is already registered in `/common/backbones/resnet.py`, so no extra
code is needed. **Transform** is also pre-registered and available in `/tasks/aigc/transforms/aigc_transforms.py`, providing basic
augmentations and ImageNet-standard normalization.

3. YAML Config & Training

ForensicHub supports lightweight configuration via YAML files. In this example, aside from data preparation, no
additional code is required.
Here is a sample training YAML `/statics/aigc/resnet_train.yaml`. The four components-**Model, Dataset, Transform, Evaluator**-are all initiated
via `init_config`：

```
# DDP
gpus: "4,5"
flag: train

# Log
log_dir: "./log/aigc_resnet_df_train"

# Task
if_predict_label: true
if_predict_mask: false

# Model
model:
  name: Resnet50
  # Model specific setting
  init_config:
    pretrained: true
    num_classes: 1

# Train dataset
train_dataset:
  name: AIGCLabelDataset
  dataset_name: DiffusionForensics_train
  init_config:
    image_size: 224
    path: /mnt/data1/public_datasets/AIGC/DiffusionForensics/images/train.json
#  Test dataset (one or many)
test_dataset:
  - name: AIGCLabelDataset
    dataset_name: DiffusionForensics_val
    init_config:
      image_size: 224
      path: /mnt/data1/public_datasets/AIGC/DiffusionForensics/images/val.json

# Transform
transform:
  name: AIGCTransform

# Evaluators
evaluator:
  - name: ImageF1
    init_config:
      threshold: 0.5

# Training related
batch_size: 768
test_batch_size: 128
epochs: 20
accum_iter: 1
record_epoch: 0  # Save the best only after record epoch.

# Test related
no_model_eval: false
test_period: 1

# Logging & TensorBoard
log_per_epoch_count: 20

# DDP & AMP settings
find_unused_parameters: false
use_amp: true

# Optimizer parameters
weight_decay: 0.05
lr: 1e-4
blr: 0.001
min_lr: 1e-5
warmup_epochs: 1

# Device and training control
device: "cuda"
seed: 42
resume: ""
start_epoch: 0
num_workers: 8
pin_mem: true

# Distributed training parameters
world_size: 1
local_rank: -1
dist_on_itp: false
dist_url: "env://"
```

After creating the YAML file, you can launch training using `statics/run.sh` after updating file paths. You can also
use`statics/batch_run.sh`
for batch experiments, which internally invokes multiple `run.sh` scripts. Testing works similarly and only requires
configuring the same four components.



</details>

## Citation

```
@misc{du2025forensichubunifiedbenchmark,
      title={ForensicHub: A Unified Benchmark & Codebase for All-Domain Fake Image Detection and Localization}, 
      author={Bo Du and Xuekang Zhu and Xiaochen Ma and Chenfan Qu and Kaiwen Feng and Zhe Yang and Chi-Man Pun and Jian Liu and Jizhe Zhou},
      year={2025},
      eprint={2505.11003},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.11003}, 
}
```
