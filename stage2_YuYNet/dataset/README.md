# Dataset README

这个文件夹用于存放 `stage2_YuYNet` 当前模型训练所需的数据。

## 目标

这里的数据是为当前的 CPP latent dynamics 基线模型准备的。模型默认读取：

- 单试次 EEG
- 刺激锁定时间轴
- `CP1`、`CP2`、`CPz` 三个通道
- 与每个试次一一对应的行为和质量说明信息

## 推荐文件结构

```text
dataset/
├── README.md
├── eeg_cpp_trials.npy
├── metadata.csv
├── times_ms.npy
├── channel_names.txt
└── preprocessing_notes.md
```

## 每个文件的含义

### `eeg_cpp_trials.npy`

这是主数据文件，保存每个试次的 EEG 数值。

- 结构顺序应为：`试次 × 时间点 × 通道`
- 通道数当前应为 3
- 通道顺序应与 `channel_names.txt` 完全一致

当前项目默认要求通道顺序为：

```text
CP1
CP2
CPz
```

### `metadata.csv`

这是每个试次对应的一行说明信息。它的行数必须和 `eeg_cpp_trials.npy` 里的试次数一致。

当前项目要求至少包含以下列：

- `subject_id`
- `trial_id`
- `condition`
- `evidence_strength`
- `choice`
- `correctness`
- `RT_ms`
- `response_hand`
- `artifact_rejection_flag`

补充说明：

- `RT_ms` 是反应时，单位是毫秒
- `response_hand` 用于记录左右手反应
- `artifact_rejection_flag` 用于标记该试次是否应被视为伪迹或无效
- 如果你原始数据里没有 `evidence_strength`，当前程序也接受把它写成 `difficulty`

### `times_ms.npy`

这是时间轴文件。

- 长度应与 `eeg_cpp_trials.npy` 里的时间点数量一致
- 单位是毫秒
- 一般应覆盖刺激前到反应前后的时间范围

### `channel_names.txt`

这是通道名文件。

- 每行一个通道名
- 顺序必须和 `eeg_cpp_trials.npy` 的最后一维完全一致
- 当前模型要求内容为：

```text
CP1
CP2
CPz
```

### `preprocessing_notes.md`

这是前处理说明文件，用来简单记录这批数据是怎么得到的。

建议至少写清楚：

- 参考方式
- 滤波设置
- 去伪迹方法
- 基线处理方式

## 使用前需要满足的检查

在训练前，这些关系需要成立：

- `metadata.csv` 的行数 = `eeg_cpp_trials.npy` 的试次数
- `times_ms.npy` 的长度 = `eeg_cpp_trials.npy` 的时间点数
- `channel_names.txt` 的通道顺序与 EEG 数据完全一致
- 元数据里必须能找到有效的反应时
- 至少有一部分试次没有被标记为伪迹

## 当前状态

这个 `README.md` 已经准备好，但正式训练数据文件需要你后续放入这个目录中。

