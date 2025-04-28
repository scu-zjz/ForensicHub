#!/bin/bash



yaml_config="/mnt/data0/xuekang/workspace/ForensicHub/ForensicHub/statics/document/trufor_train.yaml"

# 从 yaml 中读取 gpus、log_dir 和 flag
gpus=$(python -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['gpus'])")
base_dir=$(python -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['log_dir'])")
flag=$(python -c "import yaml; print(yaml.safe_load(open('$yaml_config'))['flag'])")

# 计算 GPU 数量
gpu_count=$(echo $gpus | awk -F',' '{print NF}')

# 环境变量设置
export PYTHONPATH=$(pwd)/ForensicHub:$PYTHONPATH
mkdir -p ${base_dir}

# 根据 flag 决定运行哪个脚本
if [ "$flag" = "test" ]; then
    script_path="ForensicHub/training_scripts/test.py"
elif [ "$flag" = "train" ]; then
    script_path="ForensicHub/training_scripts/train_doc.py"
else
    echo "配置文件中的 flag 字段必须是 'test' 或 'train'，当前是 '$flag'"
    exit 1
fi

# 启动
CUDA_VISIBLE_DEVICES=${gpus} \
torchrun  \
    --standalone    \
    --nnodes=1     \
    --nproc_per_node=${gpu_count} \
${script_path} \
   --config $yaml_config \
2> ${base_dir}/error.log 1>${base_dir}/logs.log
