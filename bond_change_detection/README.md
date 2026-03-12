# bond_change_detection

This module detects bond formation and bond breaking events from AIMD trajectories.

本模块用于从 AIMD 轨迹中识别成键与断键事件，并导出发生变化的关键结构帧。

## Goal

- Load frames exported by `aimd_scripts`
- Build per-frame bonding graphs from distance cutoffs
- Compare consecutive frames
- Export bond change events and candidate key structures

## 模块目标

- 读取 `aimd_scripts` 导出的轨迹帧
- 基于原子间距离 cutoff 构建每一帧的键连接关系
- 比较相邻两帧之间的键变化
- 导出成键/断键事件表以及关键结构帧

## Logic

1. Read the trajectory path and pair cutoffs from `config.yaml`
2. Load frames from a CIF directory or a multi-frame trajectory file
3. Build a bond map for each frame with ASE `neighbor_list`
4. Compare the current frame with the previous frame
5. Mark new bonds as `formed` and missing bonds as `broken`
6. Save an event table and optionally export changed frames

## 逻辑说明

1. 从 `config.yaml` 读取轨迹路径和元素对 cutoff
2. 加载轨迹帧，目前支持 `cif` 文件目录或单个多帧轨迹文件
3. 对每一帧调用 ASE 的 `neighbor_list` 构建 bond map
4. 将当前帧和前一帧的 bond map 进行比较
5. 当前帧新增的键记为 `formed`
6. 前一帧存在但当前帧消失的键记为 `broken`
7. 输出事件表，并按需保存发生变化的关键帧

## Expected inputs

- A trajectory file or a directory of frame files
- Element-pair cutoff definitions

## 输入说明

- 一个轨迹文件，或一个包含逐帧结构文件的目录
- 元素对的 cutoff 定义

对于当前项目，推荐直接使用 AIMD 结果目录中的：

```text
AIMD_dataset/cif_frames/
  frame_00000100.cif
  frame_00000200.cif
  ...
```

也就是说，本模块默认分析的是 `cif_frames`，而不是 `deepmd_dataset`。

## Expected outputs

- `outputs/bond_events.csv`
- `outputs/key_frames/`
- `outputs/summary.json`

## 输出说明

- `outputs/bond_events.csv`
  记录每一条成键或断键事件
- `outputs/key_frames/`
  保存发生键变化的关键帧结构
- `outputs/summary.json`
  记录输入路径、读入帧数、检测到的事件数和运行状态

`bond_events.csv` 主要字段包括：

- `frame_index`: 当前帧序号
- `frame_label`: 当前帧标签
- `previous_frame_label`: 前一帧标签
- `event_type`: `formed` 或 `broken`
- `atom_i`, `atom_j`: 发生变化的原子编号
- `symbol_i`, `symbol_j`: 对应元素
- `pair`: 元素对类型，例如 `C-O`、`Zn-O`
- `distance_previous`: 前一帧中的键长
- `distance_current`: 当前帧中的键长

## Minimal workflow

1. Read frames
2. Build bond list per frame
3. Diff bonds between neighboring frames
4. Save event table and selected frames

## 运行方式

在模块目录下执行：

```powershell
cd c:\Users\yingkaiwu\Desktop\Active-learning\NNP-reference\scripts\bond_change_detection
python run.py
```

## 配置说明

配置文件为 [config.yaml](./config.yaml)。

示例：

```yaml
input:
  trajectory: "../aimd_scripts/AIMD_dataset/cif_frames"
  format: "cif_dir"
```

其中：

- `trajectory` 指向 AIMD 导出的 `cif_frames` 目录
- `format: "cif_dir"` 表示输入是逐帧 CIF 文件目录

如果你改用单个多帧轨迹文件，则可以切换成：

```yaml
input:
  trajectory: "../path/to/trajectory.xyz"
  format: "trajectory_file"
```

## Notes

- The current implementation compares neighboring frames only.
- If fewer than two frames are found, the module writes an empty CSV and a `no_input` or `insufficient_frames` status.
- Cutoffs should be tuned for the chemistry of the target system.

## 备注

- 当前实现只比较相邻帧，不做跨帧长程回溯。
- 如果输入帧少于两帧，程序不会报错，而是输出空事件表，并在 `summary.json` 中写入状态说明。
- cutoff 直接影响“是否判定为成键”，需要根据体系化学环境进行调整。
