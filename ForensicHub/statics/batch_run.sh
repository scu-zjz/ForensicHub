#!/bin/bash

# 存放多个配置文件路径的列表
yaml_list=(
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/aigc/dualnet_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/aigc/hifinet_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/aigc/synthbuster_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/aigc/univfd_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/mask2label/catnet_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/mask2label/imlvit_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/mask2label/mesorch_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/mask2label/mvss_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/mask2label/pscc_test.yaml"
  "/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/mask2label/trufor_test.yaml"
  # 继续添加更多 YAML 路径
)

# run.sh 脚本路径
run_script="/mnt/data1/dubo/workspace/ForensicHub/ForensicHub/statics/run.sh"

# 遍历每个 YAML 配置
for yaml_config in "${yaml_list[@]}"; do
    echo "======================================"
    echo "Starting training with config: $yaml_config"
    echo "======================================"

    # 设置 YAML 环境变量以供 run.sh 使用
    export yaml_config="$yaml_config"

    # 调用 run.sh，出错不退出，继续下一个
    bash "$run_script"
    status=$?

    if [ $status -ne 0 ]; then
        echo "❌ Failed: $yaml_config"
    else
        echo "✅ Success: $yaml_config"
    fi

    echo ""  # 空行分隔日志输出
done
