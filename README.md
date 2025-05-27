# Project README

This repository implements a modular, configurable pipeline for protein sequence–based machine learning experiments. It uses Hydra/OmegaConf for configuration, separates concerns into distinct modules, and supports multiple ML models, hyperparameter optimization, and experiment tracking.

## Directory Structure

```text
project_root/
├── README.md                   # 项目说明文档，包含目录结构和使用说明
├── setup.py                    # 项目安装脚本，用于打包和依赖管理
├── .gitignore                  # Git 忽略文件配置
├── configs/                    # 配置文件目录，使用 Hydra/OmegaConf 管理
│   ├── base.yaml               # 通用基础配置文件
│   ├── data/                   # 数据相关配置
│   ├── model/                  # 模型相关配置
│   ├── experiment/             # 实验相关配置
│   └── logger/                 # 日志相关配置
├── amp/                        # 核心代码目录
│   ├── data/                   # 数据处理模块
│   │   ├── feature/            # 特征提取代码
│   │   └── split/              # 数据集划分代码
│   ├── models/                 # 模型实现目录
│   │   ├── ml/                 # 传统机器学习模型
│   │   │   ├── classification/ # 分类模型（如 SVM、随机森林）
│   │   │   ├── regression/     # 回归模型（如线性回归）
│   │   │   └── sequence/       # 序列模型（如 HMM、CRF）
│   │   ├── dl/                 # 深度学习模型
│   │   │   ├── classification/ # 深度学习分类模型（如 CNN、Transformer）
│   │   │   ├── regression/     # 深度学习回归模型
│   │   │   └── sequence/       # 深度学习序列模型（如 RNN、Transformer）
│   │   ├── multi_task/         # 多任务模型
│   │   │   ├── shared_layers.py # 共享层设计
│   │   │   ├── multi_task_model.py # 多任务模型主结构
│   │   │   └── task_specific.py # 任务特定的头部
│   ├── layers/                 # 通用层设计
│   │   ├── attention.py        # 注意力机制
│   │   ├── fully_connected.py  # 自定义全连接层
│   │   ├── convolutional.py    # 自定义卷积层
│   │   └── recurrent.py        # 自定义循环层
│   ├── metrics/                # 评估指标模块
│   ├── trainer/                # 模型训练模块
│   ├── postprocess/            # 后处理模块
│   └── utils/                  # 工具函数模块
├── tests/                      # 测试代码目录
│   ├── test_data.py            # 数据处理相关测试
│   ├── test_model.py           # 模型相关测试
│   └── test_utils.py           # 工具函数相关测试
├── notebooks/                  # Jupyter Notebook 目录，用于交互式实验
│   └── exploratory_analysis.ipynb # 数据探索分析
├── scripts/                    # 脚本目录
│   ├── run_all.sh              # 一键运行脚本
│   └── report_generator.py     # 报告生成脚本
```

---

### **修改说明**
1. **新增目录：**
   - `layers/`：存放通用的神经网络层设计，便于复用。
   - `multi_task/`：专门存放多任务模型的实现，包括共享层和任务特定头部。

2. **调整模型目录：**
   - 将传统机器学习模型和深度学习模型分开存放，避免混淆。
   - 深度学习模型进一步细分为分类、回归和序列模型。

3. **注释：**
   - 为每个目录和文件添加了详细的注释，说明其用途和内容。

---

### **总结**
这个更新后的 `README.md` 更加清晰地展示了项目的结构，特别是对多任务模型和自定义层的支持。你可以直接将其替换到你的项目中，方便团队成员快速理解项目结构。
---

### YAML Configuration

`amp.utils` 提供 `load_yaml_config` 函数用于读取 YAML 配置文件。
`ModelModule` 及其子类新增 `from_yaml` 方法，可以直接从 YAML
文件实例化模型。

```python
from amp.model.ml.svm import SVM

model = SVM.from_yaml("configs/model/svm.yaml")
```
