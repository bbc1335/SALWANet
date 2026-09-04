<div align="center">

# SALWANet

**Scale-Aware Label Assignment and Wavelet-Attention for UAV Small Object Detection**

[简体中文](README.md) · [English](README_EN.md)

</div>

SALWANet 是论文《SALWANet: Scale-Aware Label Assignment and Wavelet-Attention for UAV Small Object Detection》的配套目标检测仓库。项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 二次开发，面向无人机（UAV）航拍影像中的小目标与密集目标检测，保留 Ultralytics 的训练、验证和推理接口，并提供 `n/s/m/l` 四种模型规模。

## 环境与安装

项目沿用 [pyproject.toml](pyproject.toml)，Python 要求为 `>=3.8`。建议创建独立虚拟环境，再根据显卡和 CUDA 版本安装匹配的 PyTorch：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 快速开始

### 训练

仓库根目录的 [train.py](train.py) 默认使用 SALWANetn 在 VisDrone 上训练，直接运行：

```bash
python train.py
```

在 Python 中等价于：

```python
from ultralytics import SALWANet

model = SALWANet("SALWANetn.yaml")  # build a new model from scratch
model.train(
    data="VisDrone.yaml",
    epochs=300,
    batch=8,
    imgsz=640,
    workers=8,
)
```

需要依次训练多个规模时：

```python
for name in ["SALWANetn", "SALWANets", "SALWANetm", "SALWANetl"]:
    model = SALWANet(f"{name}.yaml")
    model.train(data="VisDrone.yaml", epochs=300, batch=8, imgsz=640, workers=8)
```

默认输出目录为 `runs/detect/train`，最优权重位于 `runs/detect/train/weights/best.pt`。

### 验证

仓库根目录的 [val.py](val.py) 是验证入口。将 `best.pt` 替换为实际权重路径后运行，或在 Python 中执行：

```python
from ultralytics import SALWANet

model = SALWANet("best.pt")
res = model.val(data="VisDrone.yaml", batch=8, imgsz=640, save_json=True)
```

`save_json=True` 会将预测结果保存为 COCO 格式 JSON，便于后续分析。

### 推理

本地批量推理可参考 [predict.py](predict.py)（默认被 git 忽略）；公开接口与 Ultralytics 一致：

```python
from ultralytics import SALWANet

model = SALWANet("path/to/best.pt")
model.predict(source="path/to/images/", save=True, conf=0.25, imgsz=640, line_width=2)
```

推理结果默认保存在 `runs/detect/predict`。

## 模型配置

| 模型 | YAML | 缩放系数 `[depth, width, max_channels]` |
| --- | --- | --- |
| SALWANetn | [SALWANetn.yaml](ultralytics/cfg/models/v8/SALWANetn.yaml) | `[0.33, 0.25, 1024]` |
| SALWANets | [SALWANets.yaml](ultralytics/cfg/models/v8/SALWANets.yaml) | `[0.33, 0.50, 1024]` |
| SALWANetm | [SALWANetm.yaml](ultralytics/cfg/models/v8/SALWANetm.yaml) | `[0.67, 0.75, 768]` |
| SALWANetl | [SALWANetl.yaml](ultralytics/cfg/models/v8/SALWANetl.yaml) | `[1.00, 1.00, 512]` |

## 数据集

仓库已包含 [VisDrone.yaml](ultralytics/cfg/datasets/VisDrone.yaml)，共 10 类。该 YAML 同时提供 VisDrone 官方数据的下载入口与原始标注转 YOLO 格式的脚本。

使用自定义数据集时，请先准备如下目录结构，并在 YAML 中设置 `path/train/val/names`：

```text
datasets/
└── mydata/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```

```yaml
# mydata.yaml
path: datasets/mydata
train: images/train
val: images/val

names:
  0: class_a
  1: class_b
```

然后以 `data="mydata.yaml"` 传入训练或验证接口。

## 仓库结构

```text
train.py                      训练入口（默认 SALWANetn，VisDrone）
val.py                        best.pt 验证入口
predict.py                    本地批量推理示例（默认 git 忽略）
ultralytics/cfg/models/v8/    SALWANetn/s/m/l 模型 YAML
ultralytics/cfg/datasets/     VisDrone 等数据集 YAML
ultralytics/nn/modules/       WaveletDownsample、AttnRes、C2f_AttnRes
ultralytics/nn/tasks.py       HCSYOLOModel 与自定义前向
ultralytics/utils/tal.py      NewTaskAlignedAssigner（SAL-TAL）
ultralytics/utils/loss.py     HCSYOLOLoss
ultralytics/models/yolo/      SALWANet API 与训练器
runs/                         训练输出目录，默认 git 忽略
```

## License

本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 修改，代码继续沿用仓库根目录 [LICENSE](LICENSE) 中的 AGPL-3.0 协议。使用或分发本项目代码时，请遵守 Ultralytics 原项目与本仓库的许可证条款。
