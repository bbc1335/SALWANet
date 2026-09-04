# SALWANet

SALWANet 是基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 二次开发的目标检测研究项目，面向航拍、遥感与工业场景中的小目标和密集目标检测。项目保留 Ultralytics 的训练、验证和推理接口，主要工作在 YOLOv8 骨干网络上引入小波下采样与跨阶段注意力残差融合，并提供 `n/s/m/l` 四种模型配置。

## 主要改进

- **WaveletDownsample**：用 Haar 小波四频带分解替代部分步幅卷积。输入经 `dwt_init` 得到 `LL/LH/HL/HH` 四组特征，再用 `1x1` 卷积压缩通道，同时保留一条步幅 `3x3` 卷积分支做并行残差，缓解小目标特征在下采样过程中的信息损失。
- **AttnRes**：接收当前阶段特征与已完成分辨率对齐的历史特征，对 Query、Key、Value 计算通道维相似度并做 softmax 加权融合，最后以残差形式输出。实现见 [csattnres.py](ultralytics/nn/modules/csattnres.py)。
- **C2f_AttnRes**：在 `C2f` 结构内先用 `Focus/Conv` 对齐 P2、P3 历史特征，再调用 `AttnRes` 融合。SALWANet 配置中 P4、P5 阶段开启跨阶段特征融合。
- **HCSYOLOModel**：自定义前向逻辑在特征图经过时保存早期 `C2f_AttnRes` 输出，并将历史特征列表传给后续需要融合的阶段，见 [tasks.py](ultralytics/nn/tasks.py)。
- **SALWANet API**：通过 `SALWANet("SALWANetn.yaml")` 直接构建模型，训练、验证、推理均使用 Ultralytics 原生方法，注册位置见 [model.py](ultralytics/models/yolo/model.py)。

## 使用前说明

当前 `main` 分支仍处于代码整理阶段。SALWANet 配置实际使用的自定义模块是 [csattnres.py](ultralytics/nn/modules/csattnres.py) 中的 `WaveletDownsample`、`AttnRes` 和 `C2f_AttnRes`；但 `ultralytics/nn/modules/__init__.py` 和 [tasks.py](ultralytics/nn/tasks.py) 仍引用了 `scjw/cps/sdsa/fbdm/sstm` 等旧实验模块，也引用了 `csattnres.py` 中尚未定义的 `CS_AttnRes_Neck`、`WaveletDownsample1/2`。同步补齐上述定义，或按 SALWANet 实际需要精简相关导入之后，才能直接执行下面的 `from ultralytics import SALWANet`。

## 环境与安装

项目沿用 Ultralytics 的 `pyproject.toml`，Python 版本要求为 `>=3.8`。建议创建独立虚拟环境后从仓库根目录安装：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install einops
```

依赖项以 [pyproject.toml](pyproject.toml) 为准，实际训练前请根据显卡和 CUDA 版本安装匹配的 PyTorch。`einops` 是 `csattnres.py` 中的导入项，如环境中没有会自动缺失，因此单独列出。

## 快速开始

### 训练

训练入口与 [train.py](train.py) 保持一致。下面的命令会从 YAML 配置新建模型，并训练 VisDrone 检测任务：

```python
from ultralytics import YOLO, SALWANet

model_name = ["SALWANetn"]

for name in model_name:
    model = SALWANet(f"{name}.yaml")  # build a new model from scratch
    model.train(
        data="VisDrone.yaml",
        epochs=300,
        batch=8,
        imgsz=640,
        workers=8,
        project="runs/0830VisDrone",
        name=f"VisDrone_DET_{name}_e300bs8sz640_k37",
    )
```

`SALWANetn.yaml` 会被 Ultralytics 从 `ultralytics/cfg/models/v8/` 下自动解析，也可以显式传入 [SALWANetn.yaml](ultralytics/cfg/models/v8/SALWANetn.yaml) 的完整路径。训练结果默认写入 `project/name/weights/best.pt`，例如：

```text
runs/0830VisDrone/VisDrone_DET_SALWANetn_e300bs8sz640_k37/weights/best.pt
```

更换模型时只需修改 `model_name`：

```python
model_name = ["SALWANetn", "SALWANets", "SALWANetm", "SALWANetl"]
```

### 验证

验证入口与 [val.py](val.py) 保持一致。请将 `best.pt` 换成实际训练得到的权重路径：

```python
from ultralytics import SALWANet

model = SALWANet("runs/0830VisDrone/VisDrone_DET_SALWANetn_e300bs8sz640_k37/weights/best.pt")
res = model.val(
    data="VisDrone.yaml",
    batch=8,
    imgsz=640,
    save_json=True,
)
```

`save_json=True` 会把预测结果保存为 COCO 格式 JSON，便于后续做类别相似度、误检过滤等分析。

### 推理

仓库根目录的 `predict.py` 是本地批量推理示例，仓库默认忽略该文件。公开的推理方式与 Ultralytics 一致：

```python
from ultralytics import SALWANet

model = SALWANet("path/to/best.pt")

# 单图、图片列表或目录均可作为 source
model.predict(
    source="test_AITOD/",
    save=True,
    conf=0.25,
    imgsz=800,
    line_width=2,
)
```

推理输出默认保存在 `runs/detect/predict` 或 `runs/detect/predictN` 中。

## 模型配置

四种 SALWANet 配置均基于 YOLOv8 检测结构，输出 `P3/P4/P5` 三个尺度的 `Detect` 头：

| 模型 | YAML | 缩放系数 `[depth, width, max_channels]` |
| --- | --- | --- |
| SALWANetn | [SALWANetn.yaml](ultralytics/cfg/models/v8/SALWANetn.yaml) | `[0.33, 0.25, 1024]` |
| SALWANets | [SALWANets.yaml](ultralytics/cfg/models/v8/SALWANets.yaml) | `[0.33, 0.50, 1024]` |
| SALWANetm | [SALWANetm.yaml](ultralytics/cfg/models/v8/SALWANetm.yaml) | `[0.67, 0.75, 768]` |
| SALWANetl | [SALWANetl.yaml](ultralytics/cfg/models/v8/SALWANetl.yaml) | `[1.00, 1.00, 512]` |

配置文件默认 `nc: 80`。使用自定义数据集训练时，Ultralytics 会用数据集 YAML 中的 `nc` 和 `names` 自动覆盖模型配置，不需要手工修改模型 YAML。

### 结构组成

以 [SALWANetn.yaml](ultralytics/cfg/models/v8/SALWANetn.yaml) 为例，主干部分依次为：

```text
WaveletDownsample(P1/2)
WaveletDownsample(P2/4)
C2f_AttnRes(P2，不开启跨阶段融合)
WaveletDownsample(P3/8)
C2f_AttnRes(P3，不开启跨阶段融合)
WaveletDownsample(P4/16)
C2f_AttnRes(P4，开启融合)
Conv(3x3, stride=1, 顶层语义分支)
C2f_AttnRes(P5/顶层特征，开启融合)
SPPF
C2PSA
```

颈部继续沿用 Ultralytics 的 FPN-PAN 结构（`Concat + C2f`），最后通过 `Detect` 输出 `P3/P4/P5` 预测。

### 关键文件

| 文件 | 作用 |
| --- | --- |
| [csattnres.py](ultralytics/nn/modules/csattnres.py) | `WaveletDownsample`、`AttnRes`、`C2f_AttnRes`、`C3k2_AttnRes` 等自定义模块 |
| [tasks.py](ultralytics/nn/tasks.py) | `HCSYOLOModel`、模型 YAML 解析与自定义前向逻辑 |
| [model.py](ultralytics/models/yolo/model.py) | `SALWANet` 高层 API，将 `detect` 任务映射到 `HCSYOLOModel` |
| [train.py](ultralytics/models/yolo/detect/train.py) | `HCSYOLOTrainer` 训练器 |
| [VisDrone.yaml](ultralytics/cfg/datasets/VisDrone.yaml) | VisDrone 数据集定义与下载/转换脚本 |

## 数据集

仓库提交了 [VisDrone.yaml](ultralytics/cfg/datasets/VisDrone.yaml)（10 类），该 YAML 包含下载和 VisDrone 原格式转 YOLO 格式的脚本。当前工作区还保留以下实验数据集 YAML，但它们包含本机绝对路径，并且默认被 [.gitignore](.gitignore) 忽略，使用时请改为自己的路径：

| 数据集 | 类别数 | 主要场景 |
| --- | --- | --- |
| AI-TOD | 8 | 遥感小目标 |
| DIOR | 20 | 遥感目标 |
| NEU-DET | 6 | 钢材表面缺陷 |
| GC-DET | 10 | 焊缝/工业表面缺陷 |
| BTW-DET | 4 | 工业缺陷 |

自定义数据集建议先确认目录中存在 `images/{train,val}` 和 `labels/{train,val}` 的 YOLO 格式数据，再在数据集 YAML 中设置 `path/train/val/names`。

## 仓库结构与实验脚本

### 已入库入口

```text
train.py                  VisDrone/SALWANet 训练入口
val.py                    best.pt 验证入口
ultralytics/cfg/          模型与数据集 YAML
ultralytics/models/       SALWANet API 与训练器
ultralytics/nn/           自定义模块与模型前向逻辑
runs/                     训练输出目录，默认被 git 忽略
```

### 本地实验与可视化脚本

仓库根目录还保留一批实验脚本，主要用于论文实验、分析或结果可视化。这些脚本当前大多在 `.gitignore` 中且未入库，只在本地工作区使用：

| 脚本 | 说明 |
| --- | --- |
| `predict.py` | 使用训练好的 `best.pt` 对测试目录批量推理并保存 |
| `heatmap.py` | 基于 Grad-CAM 类方法生成检测热力图 |
| `testFPS.py` | FPS 与速度基准测试 |
| `test.py` | 高斯导数显著性等图像显著性实验 |
| `analyze_class_similarity.py` | 分析类别相似度与误检 |
| `drawGT.py` | 将真实标注画到原图上 |
| `generate_filtered_annotations.py` | 根据 `predictions.json` 生成过滤后的标注 |
| `yolo2coco.py` | 将 YOLO 格式标注转换为 COCO JSON |
| `make_sci_figure.py` | 生成论文用对比图 |
| `make_waveds_spectral_figure.py` | 分析第一级下采样层的频率敏感性 |

## License

本项目基于 [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) 修改，代码继续沿用仓库根目录 [LICENSE](LICENSE) 中的 AGPL-3.0 协议。使用或分发本项目代码时，请遵守 Ultralytics 原项目的许可证条款。
