# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
# Ultralytics YOLO框架的AGPL-3.0许可证

from __future__ import annotations  # 启用类型注解的前向引用

import math  # 数学运算库
import random  # 随机数生成库
from copy import copy  # 对象复制功能
from typing import Any  # 类型注解支持

import numpy as np  # 数值计算库
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块

# 导入Ultralytics框架的相关模块
from ultralytics.data import build_dataloader, build_yolo_dataset  # 数据加载和数据集构建
from ultralytics.engine.trainer import BaseTrainer  # 基础训练器类
from ultralytics.models import yolo  # YOLO模型模块
from ultralytics.nn.tasks import DetectionModel, SSTNModel, HCSYOLOModel  # 检测模型类
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK  # 默认配置、日志记录器、进程排名
from ultralytics.utils.patches import override_configs  # 配置覆盖功能
from ultralytics.utils.plotting import plot_images, plot_labels  # 图像和标签绘制功能
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model  # PyTorch分布式训练工具


class DetectionTrainer(BaseTrainer):
    """
    基于检测模型的训练器类，继承自BaseTrainer基类。

    这个训练器专门用于目标检测任务，处理训练YOLO模型进行目标检测的特定需求，
    包括数据集构建、数据加载、预处理和模型配置。

    属性:
        model (DetectionModel): 正在训练的YOLO检测模型
        data (dict): 包含数据集信息的字典，包括类别名称和类别数量
        loss_names (tuple): 训练中使用的损失组件名称(box_loss, cls_loss, dfl_loss)

    方法:
        build_dataset: 构建用于训练或验证的YOLO数据集
        get_dataloader: 构造并返回指定模式的数据加载器
        preprocess_batch: 通过缩放和转换为浮点数来预处理图像批次
        set_model_attributes: 基于数据集信息设置模型属性
        get_model: 返回YOLO检测模型
        get_validator: 返回用于模型评估的验证器
        label_loss_items: 返回带有标记训练损失项的损失字典
        progress_string: 返回格式化训练进度字符串
        plot_training_samples: 绘制带有标注的训练样本
        plot_training_labels: 创建YOLO模型的标记训练图
        auto_batch: 基于模型内存需求计算最佳批次大小

    示例:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        初始化DetectionTrainer对象，用于训练YOLO目标检测模型。

        参数:
            cfg (dict, 可选): 包含训练参数的默认配置字典
            overrides (dict, 可选): 用于覆盖默认配置的参数字典
            _callbacks (list, 可选): 在训练期间执行的回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)  # 调用父类构造函数

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        构建用于训练或验证的YOLO数据集。

        参数:
            img_path (str): 包含图像的文件夹路径
            mode (str): 'train'模式或'val'模式，用户可以为每种模式自定义不同的数据增强
            batch (int, 可选): 批次大小，用于'rect'模式

        返回:
            (Dataset): 为指定模式配置的YOLO数据集对象
        """
        # 计算最大步长，确保至少为32像素
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        # 构建YOLO数据集，根据模式设置矩形训练选项
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        构造并返回指定模式的数据加载器。

        参数:
            dataset_path (str): 数据集路径
            batch_size (int): 每个批次的图像数量
            rank (int): 分布式训练的进程排名
            mode (str): 'train'用于训练数据加载器，'val'用于验证数据加载器

        返回:
            (DataLoader): PyTorch数据加载器对象
        """
        # 验证模式参数的有效性
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        
        # 在分布式训练中，只在排名为0的进程中初始化数据集缓存
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        # 训练模式下启用数据打乱
        shuffle = mode == "train"
        
        # 如果数据集使用矩形训练且需要打乱，发出警告并禁用打乱
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        # 构建并返回数据加载器
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,  # 验证模式使用更多工作进程
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",  # 编译模式下训练时丢弃最后不完整的批次
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        通过缩放和转换为浮点数来预处理图像批次。

        参数:
            batch (dict): 包含批次数据的字典，其中'img'键对应图像张量

        返回:
            (dict): 包含归一化图像的预处理批次
        """
        # 将批次中的所有张量移动到指定设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        
        # 图像归一化：转换为浮点数并缩放到[0,1]范围
        batch["img"] = batch["img"].float() / 255
        
        # 如果启用多尺度训练
        if self.args.multi_scale:
            imgs = batch["img"]
            # 随机生成新的图像尺寸，在0.5到1.5倍原始尺寸之间，并调整为步长的倍数
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            # 计算缩放因子
            sf = sz / max(imgs.shape[2:])
            
            # 如果缩放因子不等于1，进行图像缩放
            if sf != 1:
                # 计算新的形状尺寸，调整为步长的倍数
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                # 使用双线性插值进行图像缩放
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """基于数据集信息设置模型属性。"""
        # 注释掉的代码：用于根据检测层数量调整超参数的示例
        # Nl = de_parallel(self.model).model[-1].nl  # 检测层数量（用于缩放超参数）
        # self.args.box *= 3 / nl  # 缩放到层数
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # 缩放到类别和层数
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # 缩放到图像尺寸和层数
        
        # 设置模型属性
        self.model.nc = self.data["nc"]  # 将类别数量附加到模型
        self.model.names = self.data["names"]  # 将类别名称附加到模型
        self.model.args = self.args  # 将超参数附加到模型
        # TODO: 未来可能添加的类别权重设置
        # self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        返回YOLO检测模型。

        参数:
            cfg (str, 可选): 模型配置文件路径
            weights (str, 可选): 模型权重文件路径
            verbose (bool): 是否显示模型信息

        返回:
            (DetectionModel): YOLO检测模型
        """
        # 创建检测模型实例
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        
        # 如果提供了权重文件，加载预训练权重
        if weights:
            model.load(weights)
        
        return model

    def get_validator(self):
        """返回用于YOLO模型验证的DetectionValidator。"""
        # 设置损失名称
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        
        # 创建并返回检测验证器
        return yolo.detect.DetectionValidator(
            self.test_loader,  # 测试数据加载器
            save_dir=self.save_dir,  # 保存目录
            args=copy(self.args),  # 复制训练参数
            _callbacks=self.callbacks  # 回调函数
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """
        返回带有标记训练损失项的损失字典。

        参数:
            loss_items (list[float], 可选): 损失值列表
            prefix (str): 返回字典中键的前缀

        返回:
            (dict | list): 如果提供了loss_items，则返回标记的损失项字典，否则返回键列表
        """
        # 生成损失键名列表
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        
        if loss_items is not None:
            # 将张量转换为5位小数的浮点数
            loss_items = [round(float(x), 5) for x in loss_items]
            # 返回键值对应的字典
            return dict(zip(keys, loss_items))
        else:
            # 返回键列表
            return keys

    def progress_string(self):
        """返回包含周期、GPU内存、损失、实例数量和尺寸的格式化训练进度字符串。"""
        # 格式化进度字符串模板
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",  # 周期
            "GPU_mem",  # GPU内存使用
            *self.loss_names,  # 损失名称
            "Instances",  # 实例数量
            "Size",  # 图像尺寸
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        绘制带有标注的训练样本。

        参数:
            batch (dict[str, Any]): 包含批次数据的字典
            ni (int): 迭代次数
        """
        # 绘制训练样本图像
        plot_images(
            labels=batch,  # 批次标签
            paths=batch["im_file"],  # 图像文件路径
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存文件名
            on_plot=self.on_plot,  # 绘图回调函数
        )

    def plot_training_labels(self):
        """创建YOLO模型的标记训练图。"""
        # 从训练数据集中提取所有边界框和类别标签
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        
        # 绘制标签分布图
        plot_labels(
            boxes,  # 边界框
            cls.squeeze(),  # 类别标签（压缩维度）
            names=self.data["names"],  # 类别名称
            save_dir=self.save_dir,  # 保存目录
            on_plot=self.on_plot  # 绘图回调函数
        )

    def auto_batch(self):
        """
        通过计算模型内存占用来获取最佳批次大小。

        返回:
            (int): 最佳批次大小
        """
        # 临时禁用缓存以计算内存占用
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            # 构建训练数据集
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        
        # 计算最大目标数量，考虑马赛克增强的4倍因子
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        
        # 删除数据集以释放内存
        del train_dataset
        
        # 调用父类的自动批次大小计算方法
        return super().auto_batch(max_num_obj)
    

class SSTNTrainer(DetectionTrainer):
    """
    基于SSTN模型的训练器类，继承自DetectionTrainer基类。
    """
    def preprocess_batch(self, batch: dict) -> dict:
        """
        按batch分组处理，内存占用极低，速度接近纯广播版
        核心：只生成单图级别的小张量，不生成全量bbox的大张量
        """
        pre_batch = super().preprocess_batch(batch)
        
        batch = generate_gt_mask(pre_batch)
        # batch = generate_gaussian_mask(pre_batch)

        return batch

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        返回YOLO检测模型。

        参数:
            cfg (str, 可选): 模型配置文件路径
            weights (str, 可选): 模型权重文件路径
            verbose (bool): 是否显示模型信息

        返回:
            (DetectionModel): YOLO检测模型
        """
        # 创建检测模型实例
        model = SSTNModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        
        # 如果提供了权重文件，加载预训练权重
        if weights:
            model.load(weights)
        
        return model

    def get_validator(self):
        """返回用于YOLO模型验证的DetectionValidator。"""
        # 设置损失名称
        self.loss_names = "box_loss", "cls_loss", "dfl_loss", "self_loss"
        
        # 创建并返回检测验证器
        return yolo.detect.DetectionValidator(
            self.test_loader,  # 测试数据加载器
            save_dir=self.save_dir,  # 保存目录
            args=copy(self.args),  # 复制训练参数
            _callbacks=self.callbacks  # 回调函数
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """
        返回带有标记训练损失项的损失字典。

        参数:
            loss_items (list[float], 可选): 损失值列表
            prefix (str): 返回字典中键的前缀

        返回:
            (dict | list): 如果提供了loss_items，则返回标记的损失项字典，否则返回键列表
        """
        # 生成损失键名列表
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        
        if loss_items is not None:
            # 将张量转换为5位小数的浮点数
            loss_items = [round(float(x), 5) for x in loss_items]
            # 返回键值对应的字典
            return dict(zip(keys, loss_items))
        else:
            # 返回键列表
            return keys


class HCSYOLOTrainer(BaseTrainer):
    """
    基于HCSYOLO模型的训练器类，继承自BaseTrainer基类。

    这个训练器专门用于目标检测任务，处理训练YOLO模型进行目标检测的特定需求，
    包括数据集构建、数据加载、预处理和模型配置。

    属性:
        model (DetectionModel): 正在训练的YOLO检测模型
        data (dict): 包含数据集信息的字典，包括类别名称和类别数量
        loss_names (tuple): 训练中使用的损失组件名称(box_loss, cls_loss, dfl_loss)

    方法:
        build_dataset: 构建用于训练或验证的YOLO数据集
        get_dataloader: 构造并返回指定模式的数据加载器
        preprocess_batch: 通过缩放和转换为浮点数来预处理图像批次
        set_model_attributes: 基于数据集信息设置模型属性
        get_model: 返回YOLO检测模型
        get_validator: 返回用于模型评估的验证器
        label_loss_items: 返回带有标记训练损失项的损失字典
        progress_string: 返回格式化训练进度字符串
        plot_training_samples: 绘制带有标注的训练样本
        plot_training_labels: 创建YOLO模型的标记训练图
        auto_batch: 基于模型内存需求计算最佳批次大小

    示例:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo11n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks=None):
        """
        初始化DetectionTrainer对象，用于训练YOLO目标检测模型。

        参数:
            cfg (dict, 可选): 包含训练参数的默认配置字典
            overrides (dict, 可选): 用于覆盖默认配置的参数字典
            _callbacks (list, 可选): 在训练期间执行的回调函数列表
        """
        super().__init__(cfg, overrides, _callbacks)  # 调用父类构造函数

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """
        构建用于训练或验证的YOLO数据集。

        参数:
            img_path (str): 包含图像的文件夹路径
            mode (str): 'train'模式或'val'模式，用户可以为每种模式自定义不同的数据增强
            batch (int, 可选): 批次大小，用于'rect'模式

        返回:
            (Dataset): 为指定模式配置的YOLO数据集对象
        """
        # 计算最大步长，确保至少为32像素
        gs = max(int(unwrap_model(self.model).stride.max() if self.model else 0), 32)
        # 构建YOLO数据集，根据模式设置矩形训练选项
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """
        构造并返回指定模式的数据加载器。

        参数:
            dataset_path (str): 数据集路径
            batch_size (int): 每个批次的图像数量
            rank (int): 分布式训练的进程排名
            mode (str): 'train'用于训练数据加载器，'val'用于验证数据加载器

        返回:
            (DataLoader): PyTorch数据加载器对象
        """
        # 验证模式参数的有效性
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        
        # 在分布式训练中，只在排名为0的进程中初始化数据集缓存
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        
        # 训练模式下启用数据打乱
        shuffle = mode == "train"
        
        # 如果数据集使用矩形训练且需要打乱，发出警告并禁用打乱
        if getattr(dataset, "rect", False) and shuffle:
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        
        # 构建并返回数据加载器
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,  # 验证模式使用更多工作进程
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",  # 编译模式下训练时丢弃最后不完整的批次
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """
        通过缩放和转换为浮点数来预处理图像批次。

        参数:
            batch (dict): 包含批次数据的字典，其中'img'键对应图像张量

        返回:
            (dict): 包含归一化图像的预处理批次
        """
        # 将批次中的所有张量移动到指定设备
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        
        # 图像归一化：转换为浮点数并缩放到[0,1]范围
        batch["img"] = batch["img"].float() / 255
        
        # 如果启用多尺度训练
        if self.args.multi_scale:
            imgs = batch["img"]
            # 随机生成新的图像尺寸，在0.5到1.5倍原始尺寸之间，并调整为步长的倍数
            sz = (
                random.randrange(int(self.args.imgsz * 0.5), int(self.args.imgsz * 1.5 + self.stride))
                // self.stride
                * self.stride
            )
            # 计算缩放因子
            sf = sz / max(imgs.shape[2:])
            
            # 如果缩放因子不等于1，进行图像缩放
            if sf != 1:
                # 计算新的形状尺寸，调整为步长的倍数
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]
                # 使用双线性插值进行图像缩放
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs
        return batch

    def set_model_attributes(self):
        """基于数据集信息设置模型属性。"""
        # 注释掉的代码：用于根据检测层数量调整超参数的示例
        # Nl = de_parallel(self.model).model[-1].nl  # 检测层数量（用于缩放超参数）
        # self.args.box *= 3 / nl  # 缩放到层数
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # 缩放到类别和层数
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # 缩放到图像尺寸和层数
        
        # 设置模型属性
        self.model.nc = self.data["nc"]  # 将类别数量附加到模型
        self.model.names = self.data["names"]  # 将类别名称附加到模型
        self.model.args = self.args  # 将超参数附加到模型
        # TODO: 未来可能添加的类别权重设置
        # self.model.class_weights = labels_to_class_weights(dataset.labels, nc).to(device) * nc

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """
        返回YOLO检测模型。

        参数:
            cfg (str, 可选): 模型配置文件路径
            weights (str, 可选): 模型权重文件路径
            verbose (bool): 是否显示模型信息

        返回:
            (DetectionModel): YOLO检测模型
        """
        # 创建检测模型实例
        model = HCSYOLOModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        
        # 如果提供了权重文件，加载预训练权重
        if weights:
            model.load(weights)
        
        return model

    def get_validator(self):
        """返回用于YOLO模型验证的DetectionValidator。"""
        # 设置损失名称
        self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        
        # 创建并返回检测验证器
        return yolo.detect.DetectionValidator(
            self.test_loader,  # 测试数据加载器
            save_dir=self.save_dir,  # 保存目录
            args=copy(self.args),  # 复制训练参数
            _callbacks=self.callbacks  # 回调函数
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """
        返回带有标记训练损失项的损失字典。

        参数:
            loss_items (list[float], 可选): 损失值列表
            prefix (str): 返回字典中键的前缀

        返回:
            (dict | list): 如果提供了loss_items，则返回标记的损失项字典，否则返回键列表
        """
        # 生成损失键名列表
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        
        if loss_items is not None:
            # 将张量转换为5位小数的浮点数
            loss_items = [round(float(x), 5) for x in loss_items]
            # 返回键值对应的字典
            return dict(zip(keys, loss_items))
        else:
            # 返回键列表
            return keys

    def progress_string(self):
        """返回包含周期、GPU内存、损失、实例数量和尺寸的格式化训练进度字符串。"""
        # 格式化进度字符串模板
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",  # 周期
            "GPU_mem",  # GPU内存使用
            *self.loss_names,  # 损失名称
            "Instances",  # 实例数量
            "Size",  # 图像尺寸
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """
        绘制带有标注的训练样本。

        参数:
            batch (dict[str, Any]): 包含批次数据的字典
            ni (int): 迭代次数
        """
        # 绘制训练样本图像
        plot_images(
            labels=batch,  # 批次标签
            paths=batch["im_file"],  # 图像文件路径
            fname=self.save_dir / f"train_batch{ni}.jpg",  # 保存文件名
            on_plot=self.on_plot,  # 绘图回调函数
        )

    def plot_training_labels(self):
        """创建YOLO模型的标记训练图。"""
        # 从训练数据集中提取所有边界框和类别标签
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        
        # 绘制标签分布图
        plot_labels(
            boxes,  # 边界框
            cls.squeeze(),  # 类别标签（压缩维度）
            names=self.data["names"],  # 类别名称
            save_dir=self.save_dir,  # 保存目录
            on_plot=self.on_plot  # 绘图回调函数
        )

    def auto_batch(self):
        """
        通过计算模型内存占用来获取最佳批次大小。

        返回:
            (int): 最佳批次大小
        """
        # 临时禁用缓存以计算内存占用
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            # 构建训练数据集
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        
        # 计算最大目标数量，考虑马赛克增强的4倍因子
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4
        
        # 删除数据集以释放内存
        del train_dataset
        
        # 调用父类的自动批次大小计算方法
        return super().auto_batch(max_num_obj)


def generate_scale_complementary_gt(
    batch_targets: torch.Tensor,  # [N, 5] 格式：[batch_idx, xc_n, yc_n, w_n, h_n]
    img_size: tuple,              # 输入图像尺寸 (H, W)
    sigma_min: float = 0.25,      # 最小sigma，防止极小目标信号退化
    sigma_scale: float = 1.0 / 6.0  # 3σ原则的缩放系数
) -> torch.Tensor:
    """
    严格还原原论文核心设计 + 各向异性创新升级
    完全向量化实现，无逐目标循环，训练效率拉满
    返回：[B, 1, H, W] 尺度互补真值图 GT_scale
    """
    H, W = img_size
    device = batch_targets.device
    # 获取batch大小
    batch_size = int(batch_targets[:, 0].max().item()) + 1 if batch_targets.numel() > 0 else 1
    
    # 初始化真值图
    gt_scale = torch.zeros((batch_size, 1, H, W), dtype=torch.float32, device=device)
    if batch_targets.numel() == 0:
        return gt_scale

    # ===================== 1. 批量坐标转换与参数计算 =====================
    batch_idx = batch_targets[:, 0].long()  # 每个目标所属的batch索引 [N]
    xc_n, yc_n, w_n, h_n = batch_targets[:, 1], batch_targets[:, 2], batch_targets[:, 3], batch_targets[:, 4]
    
    # 批量转换为像素坐标 [N]
    xc = xc_n * W
    yc = yc_n * H
    w = w_n * W
    h = h_n * H

    # 批量计算各向异性sigma，严格遵循尺度自适应逻辑 [N]
    sigma_w = torch.clamp(w * sigma_scale, min=sigma_min)
    sigma_h = torch.clamp(h * sigma_scale, min=sigma_min)

    # ===================== 2. 生成全局坐标网格（向量化核心，仅生成1次） =====================
    y_grid, x_grid = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    # 扩展维度适配广播：[1, H, W]
    x_grid = x_grid.unsqueeze(0)
    y_grid = y_grid.unsqueeze(0)

    # 扩展目标参数维度，适配广播计算：[N, 1, 1]
    xc = xc.unsqueeze(-1).unsqueeze(-1)
    yc = yc.unsqueeze(-1).unsqueeze(-1)
    sigma_w = sigma_w.unsqueeze(-1).unsqueeze(-1)
    sigma_h = sigma_h.unsqueeze(-1).unsqueeze(-1)

    # ===================== 3. 向量化批量计算所有目标的高斯响应（无循环） =====================
    gaussian_response = torch.exp(
        - ((x_grid - xc) ** 2) / (2 * sigma_w ** 2)
        - ((y_grid - yc) ** 2) / (2 * sigma_h ** 2)
    )  # 输出形状 [N, H, W]，所有目标的高斯响应一次性算出

    # ===================== 4. 按batch聚合（原论文要求：逐像素相加） =====================
    for b in range(batch_size):
        # 筛选当前batch的所有目标
        target_mask = (batch_idx == b)
        if target_mask.any():
            # 按原论文要求，所有目标响应相加
            gt_scale[b, 0] = gaussian_response[target_mask].sum(dim=0)

    return gt_scale


def generate_gt_mask(batch: dict) -> dict:
    """
    按batch分组处理，内存占用极低，速度接近纯广播版
    核心：只生成单图级别的小张量，不生成全量bbox的大张量
    """        
    batch_idx = batch.get("batch_idx", None)
    bboxes = batch.get("bboxes", None)
    img = batch.get("img", None)
    
    # 快速返回空mask
    if bboxes is None or batch_idx is None or img is None or len(bboxes) == 0:
        if img is not None:
            batch_size, _, img_h, img_w = img.shape
            batch["mask"] = torch.zeros((batch_size, 1, img_h, img_w), 
                                        dtype=img.dtype, device=img.device)
        return batch
    
    # 强制类型转换
    bboxes = torch.as_tensor(bboxes, device=img.device)
    batch_idx = torch.as_tensor(batch_idx, device=img.device, dtype=torch.long)
    
    batch_size, _, img_h, img_w = img.shape
    
    # 1. 批量计算所有bbox坐标（向量化，只做1次）
    x_center, y_center, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    x1 = ((x_center - width / 2) * img_w).long()
    y1 = ((y_center - height / 2) * img_h).long()
    x2 = ((x_center + width / 2) * img_w).long()
    y2 = ((y_center + height / 2) * img_h).long()
    
    # 2. 边界裁剪+过滤无效bbox
    x1 = torch.clamp(x1, 0, img_w)
    y1 = torch.clamp(y1, 0, img_h)
    x2 = torch.clamp(x2, 0, img_w)
    y2 = torch.clamp(y2, 0, img_h)
    valid = (x2 > x1) & (y2 > y1)
    
    # 初始化最终mask
    masks = torch.zeros((batch_size, 1, img_h, img_w), 
                        dtype=img.dtype, device=img.device)
    
    if not valid.any():
        batch["mask"] = masks
        return batch
    
    # 只保留有效bbox
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    batch_idx = batch_idx[valid]
    
    # 3. 按batch分组处理（单层循环，batch_size很小，内存友好）
    # 预生成坐标网格（只生成1次，复用）
    y_coords = torch.arange(img_h, device=img.device).unsqueeze(1)  # [H, 1]
    x_coords = torch.arange(img_w, device=img.device).unsqueeze(0)  # [1, W]
    
    # 遍历唯一的batch_id（单层循环，batch_size通常<=32，可忽略性能损失）
    unique_batch_ids = torch.unique(batch_idx)
    for bid in unique_batch_ids:
        # 获取当前batch的所有bbox
        bid_mask = batch_idx == bid
        if not bid_mask.any():
            continue
        
        # 提取当前batch的bbox坐标（K=当前batch的bbox数，通常<100）
        bid_x1 = x1[bid_mask]
        bid_y1 = y1[bid_mask]
        bid_x2 = x2[bid_mask]
        bid_y2 = y2[bid_mask]
        K = len(bid_x1)
        
        # 小范围广播：生成当前batch的bbox_mask [K, H, W]（内存极小）
        bid_y1_exp = bid_y1.unsqueeze(1).unsqueeze(1)  # [K, 1, 1]
        bid_y2_exp = bid_y2.unsqueeze(1).unsqueeze(1)
        bid_x1_exp = bid_x1.unsqueeze(1).unsqueeze(1)
        bid_x2_exp = bid_x2.unsqueeze(1).unsqueeze(1)
        
        y_mask = (y_coords >= bid_y1_exp) & (y_coords < bid_y2_exp)  # [K, H, W]
        x_mask = (x_coords >= bid_x1_exp) & (x_coords < bid_x2_exp)  # [K, H, W]
        bid_bbox_mask = (y_mask & x_mask).any(dim=0)  # [H, W]（合并当前batch的所有bbox）
        
        # 赋值到最终mask
        masks[bid, 0] = bid_bbox_mask.to(img.dtype)
    
    batch["mask"] = masks
    
    return batch


def generate_gaussian_mask(batch: dict) -> dict:
    """
    生成高斯软掩码GT（融合版：内存极低+向量化高效）
    核心：按batch分组处理，只生成单图级别的小张量，不生成全量bbox的大张量
    输入batch要求：
        - "batch_idx": (num_targets,) - 每个目标对应的图片索引
        - "bboxes": (num_targets, 4) - 归一化坐标 [x_center, y_center, width, height]
        - "img": (B, 3, H, W) - 输入图片batch
    输出batch新增：
        - "mask": (B, 1, H, W) - 高斯软掩码
    """        
    batch_idx = batch.get("batch_idx", None)
    bboxes = batch.get("bboxes", None)
    img = batch.get("img", None)
    
    # -------------------------- 1. 快速返回空mask --------------------------
    if bboxes is None or batch_idx is None or img is None or len(bboxes) == 0:
        if img is not None:
            batch_size, _, img_h, img_w = img.shape
            batch["mask"] = torch.zeros((batch_size, 1, img_h, img_w), 
                                        dtype=img.dtype, device=img.device)
        return batch
    
    # -------------------------- 2. 强制类型转换与参数提取 --------------------------
    bboxes = torch.as_tensor(bboxes, device=img.device)
    batch_idx = torch.as_tensor(batch_idx, device=img.device, dtype=torch.long)
    batch_size, _, img_h, img_w = img.shape
    
    # -------------------------- 3. 批量计算所有bbox坐标与高斯参数（向量化，只做1次） --------------------------
    # 输入bbox格式: [x_center, y_center, width, height] (归一化)
    x_center, y_center, width, height = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    
    # 归一化坐标转像素坐标
    x1 = ((x_center - width / 2) * img_w).long()
    y1 = ((y_center - height / 2) * img_h).long()
    x2 = ((x_center + width / 2) * img_w).long()
    y2 = ((y_center + height / 2) * img_h).long()
    cx = (x_center * img_w).float()  # 高斯中心（像素坐标）
    cy = (y_center * img_h).float()
    w = (width * img_w).float()     # 高斯宽高（像素坐标）
    h = (height * img_h).float()
    
    # 边界裁剪
    x1 = torch.clamp(x1, 0, img_w)
    y1 = torch.clamp(y1, 0, img_h)
    x2 = torch.clamp(x2, 0, img_w)
    y2 = torch.clamp(y2, 0, img_h)
    
    # 过滤无效bbox
    valid = (x2 > x1) & (y2 > y1)
    
    # 初始化最终mask
    masks = torch.zeros((batch_size, 1, img_h, img_w), 
                        dtype=img.dtype, device=img.device)
    
    if not valid.any():
        batch["mask"] = masks
        return batch
    
    # 只保留有效bbox
    x1, y1, x2, y2 = x1[valid], y1[valid], x2[valid], y2[valid]
    cx, cy, w, h = cx[valid], cy[valid], w[valid], h[valid]
    batch_idx = batch_idx[valid]
    
    # -------------------------- 4. 预生成全局坐标网格（只生成1次，复用） --------------------------
    y_coords = torch.arange(img_h, device=img.device).float()  # [H]
    x_coords = torch.arange(img_w, device=img.device).float()  # [W]
    
    # -------------------------- 5. 按batch分组处理（单层循环，batch_size很小，内存友好） --------------------------
    # 遍历唯一的batch_id（外层循环次数<=batch_size，通常<=32，可忽略性能损失）
    unique_batch_ids = torch.unique(batch_idx)
    for bid in unique_batch_ids:
        # 获取当前batch的所有bbox
        bid_mask = batch_idx == bid
        if not bid_mask.any():
            continue
        
        # 提取当前batch的bbox参数（K=当前batch的bbox数，通常<100，内存极小）
        bid_x1 = x1[bid_mask]
        bid_y1 = y1[bid_mask]
        bid_x2 = x2[bid_mask]
        bid_y2 = y2[bid_mask]
        bid_cx = cx[bid_mask]
        bid_cy = cy[bid_mask]
        bid_w = w[bid_mask]
        bid_h = h[bid_mask]
        K = len(bid_x1)
        
        # -------------------------- 5.1 小范围广播：生成当前batch的高斯热图 --------------------------
        # 扩展参数维度以支持广播: [K] -> [K, 1, 1]
        bid_cx_exp = bid_cx.unsqueeze(1).unsqueeze(1)
        bid_cy_exp = bid_cy.unsqueeze(1).unsqueeze(1)
        bid_sigma_x_exp = (bid_w / 6.0).unsqueeze(1).unsqueeze(1)  # 高斯sigma与目标尺寸正相关
        bid_sigma_y_exp = (bid_h / 6.0).unsqueeze(1).unsqueeze(1)
        bid_x1_exp = bid_x1.unsqueeze(1).unsqueeze(1)
        bid_y1_exp = bid_y1.unsqueeze(1).unsqueeze(1)
        bid_x2_exp = bid_x2.unsqueeze(1).unsqueeze(1)
        bid_y2_exp = bid_y2.unsqueeze(1).unsqueeze(1)
        
        # 扩展坐标网格维度: [H] -> [K, H, 1], [W] -> [K, 1, W]
        # 利用广播机制自动扩展为 [K, H, W]
        y_grid_b = y_coords.unsqueeze(0).unsqueeze(2).expand(K, img_h, 1)
        x_grid_b = x_coords.unsqueeze(0).unsqueeze(1).expand(K, 1, img_w)
        
        # 向量化计算高斯分布（一次性计算当前batch所有K个目标的高斯值）
        gaussian_b = torch.exp(
            -((x_grid_b - bid_cx_exp) ** 2 / (2 * bid_sigma_x_exp ** 2 + 1e-6) +
              (y_grid_b - bid_cy_exp) ** 2 / (2 * bid_sigma_y_exp ** 2 + 1e-6))
        )  # [K, H, W]
        
        # -------------------------- 5.2 向量化生成bbox掩码（只保留bbox内的值） --------------------------
        bbox_mask_b = (x_grid_b >= bid_x1_exp) & (x_grid_b < bid_x2_exp) & \
                      (y_grid_b >= bid_y1_exp) & (y_grid_b < bid_y2_exp)  # [K, H, W]
        gaussian_b = gaussian_b * bbox_mask_b.float()  # [K, H, W]
        
        # -------------------------- 5.3 合并当前batch的所有目标（取max） --------------------------
        # 若同一位置有多个目标，取高斯值最大的那个
        merged_gaussian = gaussian_b.max(dim=0)[0]  # [H, W]
        
        # -------------------------- 5.4 赋值到最终mask --------------------------
        masks[bid, 0] = merged_gaussian.to(img.dtype)
    
    batch["mask"] = masks
    
    return batch