# Active Learning Workflow For AIMD Structure Screening

本项目用于从 AIMD 轨迹中筛选关键结构，并逐步构建用于神经网络势函数训练的高价值数据集。


## 项目目标

这套脚本围绕以下流程展开：

```text
AIMD trajectory
  -> Bond change detection
  -> Zn-O coordination change
  -> SOAP descriptor
  -> Neural network structure selector
  -> vdW energy predictor
  -> Final training dataset
```

目标是从大量 AIMD 帧中自动识别真正值得保留的结构，而不是把所有轨迹帧都直接送入训练集。

## 模块结构

- [aimd_scripts](/c:/Users/yingkaiwu/Desktop/Active-learning/NNP-reference/scripts/aimd_scripts)
  负责生成 AIMD 轨迹，并导出：
  - `deepmd_dataset/`
  - `cif_frames/`

- [bond_change_detection](/c:/Users/yingkaiwu/Desktop/Active-learning/NNP-reference/scripts/bond_change_detection)
  识别相邻帧之间的成键/断键事件，输出关键变化帧。

- [zn_o_coordination](/c:/Users/yingkaiwu/Desktop/Active-learning/NNP-reference/scripts/zn_o_coordination)
  分析 Zn-O 配位变化，用于识别吸附、脱附和表面重构相关结构。

- [soap_descriptors](/c:/Users/yingkaiwu/Desktop/Active-learning/NNP-reference/scripts/soap_descriptors)
  将结构帧转换为 SOAP 特征，作为后续 PCA、聚类和神经网络筛选的输入。

- [nn_uncertainty](/c:/Users/yingkaiwu/Desktop/Active-learning/NNP-reference/scripts/nn_uncertainty)
  当前已重定义为“神经网络结构筛选器”，而不是传统 committee uncertainty 模块。
  它使用 SOAP 特征和上游规则标签，训练一个 MLP 对结构进行打分。

- [vdw_energy_predictor](/c:/Users/yingkaiwu/Desktop/Active-learning/NNP-reference/scripts/vdw_energy_predictor)
  预留用于筛选 vdW 作用显著的结构。

- [final_training_dataset](/c:/Users/yingkaiwu/Desktop/Active-learning/NNP-reference/scripts/final_training_dataset)
  汇总多个模块筛出的候选结构，去重并整理为最终训练数据集。

## 当前推荐使用顺序

1. 运行 `aimd_scripts`，生成 `AIMD_dataset/cif_frames`
2. 运行 `bond_change_detection`，得到成键/断键事件表
3. 运行 `zn_o_coordination`，得到配位变化事件表
4. 运行 `soap_descriptors`，生成 `descriptors.npy` 和 `frame_index.csv`
5. 运行 `nn_uncertainty`，训练结构筛选神经网络并输出高分结构
6. 结合 `vdw_energy_predictor` 和其他规则模块结果
7. 运行 `final_training_dataset`，生成最终训练集

## 当前实现状态

- `aimd_scripts`
  已有可运行脚本

- `bond_change_detection`
  已实现第一版真实逻辑，可读取 `cif_frames` 并输出成键/断键事件

- `soap_descriptors`
  已完成模块说明和配置骨架，真实 descriptor 生成逻辑待实现

- `nn_uncertainty`
  已完成第一版神经网络结构筛选器框架，可读取 SOAP 特征、构造伪标签并训练简单 MLP

- 其余模块
  当前仍为骨架，已定义目录职责、配置格式和输出接口

## 输入数据约定

当前默认 AIMD 输出目录结构为：

```text
AIMD_dataset
├── deepmd_dataset
│   ├── type.raw
│   ├── type_map.raw
│   ├── set.000
│   ├── set.001
│   └── ...
└── cif_frames
    ├── frame_00000100.cif
    ├── frame_00000200.cif
    └── ...
```

其中：

- `deepmd_dataset` 主要用于势函数训练
- `cif_frames` 主要用于结构分析与筛选

## 运行方式

每个模块均在各自目录下运行，例如：

```powershell
cd c:\Users\yingkaiwu\Desktop\Active-learning\NNP-reference\scripts\bond_change_detection
python run.py
```

建议在运行前先检查对应模块的 `config.yaml`，确认输入路径与当前数据目录一致。

## 说明

- 各模块尽量保持独立，便于单独测试和后续替换实现
- 当前项目重点是先打通主动学习筛选流程，而不是一次性做复杂模型
- 神经网络模块当前采用的是“结构筛选器”思路，而不是直接预测势能或力
