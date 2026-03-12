# final_training_dataset

This module merges outputs from the earlier filters into a final training dataset.

本模块用于汇总前面多个筛选模块得到的候选结构，完成去重、来源记录和训练集划分，最终生成可用于后续势函数训练的数据集。

## Goal

- Collect candidate structures from all selection modules
- Remove duplicates
- Track provenance
- Export train, validation, and test splits

## 模块目标

- 收集各筛选模块输出的候选结构
- 去除重复样本
- 记录每个样本来自哪些筛选路径
- 导出训练集、验证集和测试集

## Logic

1. Load candidate outputs from upstream modules
2. Normalize sample identifiers
3. Merge candidate lists into one pool
4. Deduplicate repeated frames
5. Record which modules selected each frame
6. Split the final pool into train, val, and test sets
7. Export manifest files and dataset folders

## 逻辑说明

1. 读取上游模块输出的候选结果
2. 将不同模块中的样本编号统一化
3. 合并所有候选结构形成一个候选池
4. 对重复帧进行去重
5. 为每个最终样本记录来源信息 provenance
6. 按设定比例划分 train / val / test
7. 导出清单文件与数据集目录

## Expected inputs

- Candidate frame lists from:
  - bond change detection
  - Zn-O coordination analysis
  - SOAP novelty filtering
  - NN uncertainty
  - vdW predictor

## 输入说明

本模块原则上不直接处理原始 AIMD 轨迹，而是读取前几个模块的输出结果。

当前配置中预留的输入包括：

- `bond_change_detection` 的事件表
- `zn_o_coordination` 的配位变化事件表
- `soap_descriptors` 的特征元信息
- `nn_uncertainty` 的不确定性评分表
- `vdw_energy_predictor` 的 vdW 评分表

对应配置见 [config.yaml](./config.yaml)。

## Expected outputs

- `outputs/final_candidates.csv`
- `outputs/provenance.json`
- `outputs/train/`
- `outputs/val/`
- `outputs/test/`

## 输出说明

- `outputs/final_candidates.csv`
  记录最终保留的样本清单
- `outputs/provenance.json`
  记录每个样本来自哪些筛选模块
- `outputs/train/`
  训练集目录
- `outputs/val/`
  验证集目录
- `outputs/test/`
  测试集目录

后续如果需要，这里还可以继续扩展为导出 DeepMD 或其他神经网络势函数训练格式。

## Minimal workflow

1. Load candidate manifests
2. Merge and deduplicate
3. Split dataset
4. Export target training format

## 运行方式

在模块目录下执行：

```powershell
cd c:\Users\yingkaiwu\Desktop\Active-learning\NNP-reference\scripts\final_training_dataset
python run.py
```

## 配置说明

配置文件为 [config.yaml](./config.yaml)。

当前默认配置示例：

```yaml
inputs:
  bond_events: "../bond_change_detection/outputs/bond_events.csv"
  coordination_events: "../zn_o_coordination/outputs/coordination_events.csv"
  soap_metadata: "../soap_descriptors/outputs/metadata.json"
  uncertainty_scores: "../nn_uncertainty/outputs/uncertainty_scores.csv"
  vdw_scores: "../vdw_energy_predictor/outputs/vdw_scores.csv"
```

数据集划分比例由以下字段控制：

```yaml
dataset:
  split_ratio:
    train: 0.8
    val: 0.1
    test: 0.1
```

## Current status

- The folder structure is ready
- The output directories are created by `run.py`
- The full merge and deduplication logic is not implemented yet

## 当前状态

- 当前目录结构已经建立
- `run.py` 已能生成基础输出目录
- 真实的候选样本合并、去重、来源记录和数据集划分逻辑还没有正式实现

## Notes

- This module is the final aggregation stage of the workflow.
- It should use a stable sample key such as `frame_label`, `frame_index`, or structure path.
- Provenance is important because one frame may be selected by multiple criteria.

## 备注

- 该模块是整条主动学习流程的最终汇总层。
- 实现时需要选定统一的样本主键，例如 `frame_label`、`frame_index` 或结构路径。
- provenance 很重要，因为同一个结构可能同时被多个模块选中。
