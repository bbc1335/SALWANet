# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo11n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import time
import warnings
from copy import copy, deepcopy
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics import __version__
from ultralytics.cfg import get_cfg, get_save_dir
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.utils import (
    DEFAULT_CFG,
    GIT,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import ddp_cleanup, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    convert_optimizer_state_dict_to_fp16,
    init_seeds,
    one_cycle,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)


class BaseTrainer:
    """
    A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (amp.GradScaler): Gradient scaler for AMP.
        data (str): Path to data.
        ema (nn.Module): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (nn.Module): Loss function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (float): Current loss value.
        tloss (float): Total loss value.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the test set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """
        Initialize the BaseTrainer class.

        Args:
            cfg (str, optional): Path to a configuration file.
            overrides (dict, optional): Configuration overrides.
            _callbacks (list, optional): List of callback functions.
        """
        self.hub_session = overrides.pop("session", None)  # HUB
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.device = select_device(self.args.device)
        # Update "-1" devices so post-training val does not repeat search
        self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            YAML.save(self.save_dir / "args.yaml", vars(self.args))  # save run args
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo11n -> yolo11n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0

        # Callbacks
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if isinstance(self.args.device, str) and len(self.args.device):  # i.e. device='0' or device='0,1,2,3'
            world_size = len(self.args.device.split(","))
        elif isinstance(self.args.device, (tuple, list)):  # i.e. device=[0, 1, 2, 3] (multi-GPU from CLI is list)
            world_size = len(self.args.device)
        elif self.args.device in {"cpu", "mps"}:  # i.e. device='cpu' or 'mps'
            world_size = 0
        elif torch.cuda.is_available():  # i.e. device=None or device='' or device=number
            world_size = 1  # default to device 0
        else:  # i.e. device=None or device=''
            world_size = 0

        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        # Run subprocess if DDP training, else train normally
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            # Start console logging immediately at trainer initialization
            self.run_callbacks("on_pretrain_routine_start")

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Allow device='', device=None on Multi-GPU systems to default to device=0."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = generate_ddp_command(self)
            try:
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True)
            except Exception as e:
                raise e
            finally:
                ddp_cleanup(self, str(file))

        else:
            self._do_train()

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        torch.cuda.set_device(RANK)
        self.device = torch.device("cuda", RANK)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _setup_train(self):
        """在正确的进程上构建数据加载器和优化器，配置完整的训练环境。"""
        # 加载模型检查点
        ckpt = self.setup_model()
        
        # 将模型移动到指定设备（CPU/GPU）
        self.model = self.model.to(self.device)
        
        # 设置模型属性（如类别数量、锚点等）
        self.set_model_attributes()

        # 编译模型以提高性能（如果支持）
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        # 冻结层配置
        # 处理冻结参数：如果是整数则转换为范围，否则保持列表或空列表
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        
        # 总是需要冻结的层名称（目前只有.dfl层）
        always_freeze_names = [".dfl"]  # always freeze these layers
        
        # 合并所有需要冻结的层名称模式
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        self.freeze_layer_names = freeze_layer_names
        
        # 遍历模型所有参数并根据配置冻结
        for k, v in self.model.named_parameters():
            # v.register_hook(lambda x: torch.nan_to_num(x))  # 注释掉的代码：将NaN转换为0（可能导致训练不稳定）
            
            # 如果参数名称包含任何需要冻结的模式
            if any(x in k for x in freeze_layer_names):
                LOGGER.info(f"冻结层 '{k}'")
                v.requires_grad = False
            # 如果参数是浮点型但requires_grad为False，则发出警告并设置为True
            elif not v.requires_grad and v.dtype.is_floating_point:
                LOGGER.warning(
                    f"为冻结层 '{k}' 设置 'requires_grad=True'。"
                    "请参考ultralytics.engine.trainer以自定义冻结层。"
                )
                v.requires_grad = True

        # 检查并配置AMP（自动混合精度训练）
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True或False
        
        # 在单GPU或DDP的rank 0上检查AMP支持
        if self.amp and RANK in {-1, 0}:
            callbacks_backup = callbacks.default_callbacks.copy()  # 备份回调，因为check_amp()会重置它们
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # 恢复回调
        
        # DDP环境下，从rank 0广播AMP设置到所有其他rank
        if RANK > -1 and self.world_size > 1:
            dist.broadcast(self.amp.int(), src=0)  # 使用int类型广播，避免gloo后端的布尔类型错误
        
        self.amp = bool(self.amp)  # 转换为布尔类型
        # 根据PyTorch版本创建GradScaler
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        
        # DDP环境下，将模型包装为DistributedDataParallel
        if self.world_size > 1:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[RANK], find_unused_parameters=True)

        # 检查并调整输入图像大小
        # 获取模型的最大步长（用于网格大小计算）
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)
        # 确保图像大小是步长的倍数
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # 用于多尺度训练

        # 批处理大小配置
        if self.batch_size < 1 and RANK == -1:  # 仅单GPU模式下，自动估算最佳批处理大小
            self.args.batch = self.batch_size = self.auto_batch()

        # 构建数据加载器
        # 计算每个进程的批处理大小
        batch_size = self.batch_size // max(self.world_size, 1)
        
        # 创建训练数据加载器
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        
        # 创建验证数据加载器（OBB任务保持相同批大小，其他任务批大小翻倍）
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task == "obb" else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        
        # 获取验证器实例
        self.validator = self.get_validator()
        
        # 初始化模型EMA（指数移动平均）
        self.ema = ModelEMA(self.model)
        
        # 在主进程上设置指标和绘制训练标签
        if RANK in {-1, 0}:
            # 合并验证指标和损失指标键
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            
            # 如果需要绘制标签分布
            if self.args.plots:
                self.plot_training_labels()

        # 构建优化器
        # 计算梯度累积步数：确保总有效批量大小接近nbs（标称批量大小）
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)
        # 按比例缩放权重衰减
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs
        # 计算总迭代次数
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        
        # 构建优化器
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        
        # 设置学习率调度器
        self._setup_scheduler()
        
        # 初始化早停机制
        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        
        # 恢复训练（如果有检查点）
        self.resume_training(ckpt)
        
        # 设置调度器的起始迭代次数
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        
        # 调用训练前准备完成的回调
        self.run_callbacks("on_pretrain_routine_end")

    def _do_train(self):
        """
        执行模型训练的主循环，包含epoch迭代、batch处理、损失计算、优化器更新等完整训练流程。
        """
        if self.world_size > 1:
            self._setup_ddp()  # 如果使用分布式训练，设置DDP环境
        self._setup_train()  # 设置训练环境，包括数据加载器、优化器等

        nb = len(self.train_loader)  # 获取训练数据的批次数
        # 计算warmup迭代次数：如果设置了warmup_epochs，则为warmup_epochs * 批次数，否则为-1
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
        last_opt_step = -1  # 记录上一次优化器更新的迭代步数
        self.epoch_time = None  # 记录每个epoch的训练时间
        self.epoch_time_start = time.time()  # 记录训练开始时间
        self.train_time_start = time.time()  # 记录训练总时间开始时间
        self.run_callbacks("on_train_start")  # 运行训练开始回调

        # 记录训练配置信息
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )

        # 如果设置了关闭mosaic增强，计算关闭mosaic的索引位置
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        
        epoch = self.start_epoch  # 从指定的起始epoch开始训练
        self.optimizer.zero_grad()  # 清除可能残留的梯度，确保训练开始时的稳定性

        # 训练主循环
        while True:
            self.epoch = epoch  # 设置当前epoch
            self.run_callbacks("on_train_epoch_start")  # 运行epoch开始回调

            # 忽略"Detected lr_scheduler.step() before optimizer.step()"警告
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.scheduler.step()  # 更新学习率调度器

            self._model_train()  # 设置模型为训练模式
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)  # 设置DDP采样器的epoch，确保数据打乱的一致性

            pbar = enumerate(self.train_loader)  # 获取训练数据迭代器
            # 当epoch达到关闭mosaic的 epoch时，更新数据加载器的mosaic增强配置
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()  # 关闭mosaic增强
                self.train_loader.reset()  # 重置数据加载器

            # 在主进程中初始化进度条
            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())  # 记录训练进度字符串
                pbar = TQDM(enumerate(self.train_loader), total=nb)  # 使用TQDM显示进度条
            
            self.tloss = None  # 初始化总损失
            for i, batch in pbar:  # 遍历每个batch
                self.run_callbacks("on_train_batch_start")  # 运行batch开始回调
                
                # Warmup学习率和动量调整
                ni = i + nb * epoch  # 当前总的迭代步数
                if ni <= nw:
                    xi = [0, nw]  # x插值区间
                    # 计算梯度累积步数：从1线性增加到 nbs/batch_size
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for j, x in enumerate(self.optimizer.param_groups):
                        # 偏置项学习率从0.1降到lr0，其他参数学习率从0.0升到lr0
                        x["lr"] = np.interp(
                            ni, xi, [self.args.warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * self.lf(epoch)]
                        )
                        # 动量从warmup_momentum升到args.momentum
                        if "momentum" in x:
                            x["momentum"] = np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum])

                # 前向传播计算损失
                with autocast(self.amp):  # 自动混合精度训练
                    batch = self.preprocess_batch(batch)  # 预处理batch数据
                    if self.args.compile:
                        # 编译模式下：分离推理和损失计算以提高性能
                        preds = self.model(batch["img"])  # 模型推理得到预测结果
                        loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)  # 计算损失
                    else:
                        # 非编译模式下：直接调用模型得到损失
                        loss, self.loss_items = self.model(batch)
                    
                    self.loss = loss.sum()  # 计算总损失
                    if RANK != -1:
                        self.loss *= self.world_size  # 分布式训练时，损失需要乘以world_size
                    
                    # 更新总损失（加权平均）
                    self.tloss = self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)

                # 反向传播
                self.scaler.scale(self.loss).backward()  # 使用梯度缩放进行反向传播
                
                # 当达到累积步数时，更新优化器
                if ni - last_opt_step >= self.accumulate:
                    self.optimizer_step()  # 更新优化器参数
                    last_opt_step = ni  # 记录当前优化器更新的步数

                    # 如果设置了训练时间限制，检查是否超时
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # 分布式训练时，广播stop状态
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)
                            self.stop = broadcast_list[0]
                        if self.stop:  # 训练时间超过限制，停止训练
                            break

                # 日志记录
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    # 更新进度条显示
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",  # 当前epoch/总epoch
                            f"{self._get_memory():.3g}G",  # GPU内存使用情况
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # 损失值
                            batch["cls"].shape[0],  # batch大小
                            batch["img"].shape[-1],  # 图像大小
                        )
                    )
                    self.run_callbacks("on_batch_end")  # 运行batch结束回调
                    
                    # 如果当前迭代步数在plot_idx中，绘制训练样本
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")  # 运行训练batch结束回调

            # 记录学习率，用于日志记录
            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}

            self.run_callbacks("on_train_epoch_end")  # 运行epoch结束回调
            
            # 在主进程中更新EMA模型
            if RANK in {-1, 0}:
                final_epoch = epoch + 1 >= self.epochs  # 判断是否为最后一个epoch
                # 更新EMA模型的属性
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # 验证阶段：在指定条件下进行验证
            if self.args.val or final_epoch or self.stopper.possible_stop or self.stop:
                self._clear_memory(threshold=0.5)  # 清除内存，防止验证时VRAM峰值
                self.metrics, self.fitness = self.validate()  # 执行验证，获取指标和适应度值

            # 处理NaN值恢复
            if self._handle_nan_recovery(epoch):
                continue  # 如果发生NaN恢复，跳过当前epoch的后续处理

            self.nan_recovery_attempts = 0  # 重置NaN恢复尝试次数
            if RANK in {-1, 0}:
                # 保存训练指标
                self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                # 判断是否停止训练：早停条件满足或达到最后一个epoch
                self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                # 如果设置了训练时间限制，检查是否超时
                if self.args.time:
                    self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                # 保存模型：在指定条件下保存模型
                if self.args.save or final_epoch:
                    self.save_model()  # 保存模型
                    self.run_callbacks("on_model_save")  # 运行模型保存回调

            # 学习率调度和时间调整
            t = time.time()
            self.epoch_time = t - self.epoch_time_start  # 记录当前epoch的训练时间
            self.epoch_time_start = t  # 重置epoch开始时间
            if self.args.time:
                # 根据平均epoch时间调整总epoch数
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()  # 重新设置学习率调度器
                self.scheduler.last_epoch = self.epoch  # 保持调度器的last_epoch不变
                self.stop |= epoch >= self.epochs  # 如果超过了计算的总epoch数，停止训练
            
            self.run_callbacks("on_fit_epoch_end")  # 运行fit epoch结束回调
            self._clear_memory(0.5)  # 如果内存利用率超过50%，清除内存

            # 分布式训练时，广播stop状态
            if RANK != -1:
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)
                self.stop = broadcast_list[0]
            
            # 如果满足停止条件，退出训练循环
            if self.stop:
                break  # 必须中断所有DDP进程的训练循环
            epoch += 1  # 进入下一个epoch

        # 计算并记录总训练时间
        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        
        # 使用best.pt进行最终验证
        self.final_eval()
        
        if RANK in {-1, 0}:
            # 如果设置了绘图，绘制训练指标
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")  # 运行训练结束回调
        
        self._clear_memory()  # 清除所有内存
        unset_deterministic()  # 取消确定性模式
        self.run_callbacks("teardown")  # 运行清理回调

    def auto_batch(self, max_num_obj=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        return check_train_batch_size(
            model=self.model,
            imgsz=self.args.imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using polars."""
        import polars as pl  # scope for faster 'import ultralytics'

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save model training checkpoints with additional metadata."""
        import io

        # Serialize ckpt to a byte buffer once (faster than repeated torch.save() calls)
        buffer = io.BytesIO()
        torch.save(
            {
                "epoch": self.epoch,
                "best_fitness": self.best_fitness,
                "model": None,  # resume and final checkpoints derive from EMA
                "ema": deepcopy(unwrap_model(self.ema.ema)).half(),
                "updates": self.ema.updates,
                "optimizer": convert_optimizer_state_dict_to_fp16(deepcopy(self.optimizer.state_dict())),
                "scaler": self.scaler.state_dict(),
                "train_args": vars(self.args),  # save as dict
                "train_metrics": {**self.metrics, **{"fitness": self.fitness}},
                "train_results": self.read_results_csv(),
                "date": datetime.now().isoformat(),
                "version": __version__,
                "git": {
                    "root": str(GIT.root),
                    "branch": GIT.branch,
                    "commit": GIT.commit,
                    "origin": GIT.origin,
                },
                "license": "AGPL-3.0 (https://ultralytics.com/license)",
                "docs": "https://docs.ultralytics.com",
            },
            buffer,
        )
        serialized_ckpt = buffer.getvalue()  # get the serialized content to save

        # Save checkpoints
        self.wdir.mkdir(parents=True, exist_ok=True)  # ensure weights directory exists
        self.last.write_bytes(serialized_ckpt)  # save last.pt
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)  # save best.pt
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'

    def get_dataset(self):
        """
        Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] == "ndjson":
                # Convert NDJSON to YOLO format
                import asyncio

                from ultralytics.data.converter import convert_ndjson_to_yolo

                yaml_path = asyncio.run(convert_ndjson_to_yolo(self.args.data))
                self.args.data = str(yaml_path)
                data = check_det_dataset(self.args.data)
            elif self.args.data.rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """
        Load, create, or download model for any task.

        Returns:
            (dict): Optional checkpoint to resume training from.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        elif isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK == -1)  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)

    def preprocess_batch(self, batch):
        """Allow custom preprocessing model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """
        Run validation on val set using self.validator.

        Returns:
            metrics (dict): Dictionary of validation metrics.
            fitness (float): Fitness score for the validation.
        """
        if self.ema and self.world_size > 1:
            # Sync EMA buffers from rank 0 to all ranks
            for buffer in self.ema.ema.buffers():
                dist.broadcast(buffer, src=0)
        metrics = self.validator(self)
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if not self.best_fitness or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Return a NotImplementedError when the get_validator function is called."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Return dataloader derived from torch.data.Dataloader."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """
        Return a loss dict with labelled training loss items tensor.

        Note:
            This is not needed for classification but necessary for segmentation & detection
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else (("%s," * n % tuple(["epoch", "time", *keys])).rstrip(",") + "\n")  # header
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % tuple([self.epoch + 1, t, *vals])).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for object detection YOLO model."""
        model = self.best if self.best.exists() else None
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if model:
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        if model:
            LOGGER.info(f"\nValidating {model}...")
            self.validator.args.plots = self.args.plots
            self.validator.args.compile = False  # disable final val compile as too slow
            self.metrics = self.validator(model=model)
            self.metrics.pop("fitness", None)
            self.run_callbacks("on_fit_epoch_end")

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())

                # Check that resume data YAML exists, otherwise strip to force re-download of dataset
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
        if ckpt.get("optimizer") is not None:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            self.ema.ema.load_state_dict(ckpt["ema"].float().state_dict())
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness", 0.0)

    def _handle_nan_recovery(self, epoch):
        """Detect and recover from NaN/Inf loss and fitness collapse by loading last checkpoint."""
        loss_nan = self.loss is not None and not self.loss.isfinite()
        fitness_nan = self.fitness is not None and not np.isfinite(self.fitness)
        fitness_collapse = self.best_fitness and self.best_fitness > 0 and self.fitness == 0
        corrupted = RANK in {-1, 0} and loss_nan and (fitness_nan or fitness_collapse)
        reason = "Loss NaN/Inf" if loss_nan else "Fitness NaN/Inf" if fitness_nan else "Fitness collapse"
        if RANK != -1:  # DDP: broadcast to all ranks
            broadcast_list = [corrupted if RANK == 0 else None]
            dist.broadcast_object_list(broadcast_list, 0)
            corrupted = broadcast_list[0]
        if not corrupted:
            return False
        if epoch == self.start_epoch or not self.last.exists():
            LOGGER.warning(f"{reason} detected but can not recover from last.pt...")
            return False  # Cannot recover on first epoch, let training continue
        self.nan_recovery_attempts += 1
        if self.nan_recovery_attempts > 3:
            raise RuntimeError(f"Training failed: NaN persisted for {self.nan_recovery_attempts} epochs")
        LOGGER.warning(f"{reason} detected (attempt {self.nan_recovery_attempts}/3), recovering from last.pt...")
        self._model_train()  # set model to train mode before loading checkpoint to avoid inference tensor errors
        _, ckpt = load_checkpoint(self.last)
        ema_state = ckpt["ema"].float().state_dict()
        if not all(torch.isfinite(v).all() for v in ema_state.values() if isinstance(v, torch.Tensor)):
            raise RuntimeError(f"Checkpoint {self.last} is corrupted with NaN/Inf weights")
        unwrap_model(self.model).load_state_dict(ema_state)  # Load EMA weights into model
        self._load_checkpoint_state(ckpt)  # Load optimizer/scaler/EMA/best_fitness
        del ckpt, ema_state
        self.scheduler.last_epoch = epoch - 1
        return True

    def resume_training(self, ckpt):
        """Resume YOLO training from given epoch and best fitness."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert start_epoch > 0, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self._load_checkpoint_state(ckpt)
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """
        Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected
                based on the number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if
                name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [], [], []  # optimizer parameter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            name, lr, momentum = ("SGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        for module_name, module in model.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if "bias" in fullname:  # bias (no decay)
                    g[2].append(param)
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1].append(param)
                else:  # weight (with decay)
                    g[0].append(param)

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(name.lower())
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optimizer = getattr(optim, name, optim.Adam)(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optimizer = optim.RMSprop(g[2], lr=lr, momentum=momentum)
        elif name == "SGD":
            optimizer = optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for addition optimizers at https://github.com/ultralytics/ultralytics."
            )

        optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
        optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{len(g[1])} weight(decay=0.0), {len(g[0])} weight(decay={decay}), {len(g[2])} bias(decay=0.0)"
        )
        return optimizer