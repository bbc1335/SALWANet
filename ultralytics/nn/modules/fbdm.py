from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules.conv import Conv  # 假设沿用YOLO的Conv模块

class FBDM(nn.Module):
    """
    Foreground-Aware Dynamic Modulation (FBDM) module.
    
    Learns a foreground probability map and uses it to guide both spatial and 
    channel-wise modulation of the input features. The foreground map is 
    supervised by a combination of Dice loss and Focal loss, which are 
    particularly effective for handling class imbalance in foreground/background 
    segmentation. An optional consistency loss further enhances feature 
    compactness within foreground regions.
    """
    def __init__(self, c1: int, c2: int, use_consistency: bool = False, 
                 lambda_fg: float = 2.0, lambda_consis: float = 0.1):
        """
        Args:
            c1: input channels
            c2: output channels (must equal c1 for residual connection)
            use_consistency: whether to apply foreground consistency loss
            lambda_fg: weight for foreground losses (Dice + Focal)
            lambda_consis: weight for consistency loss
        """
        super().__init__()
        assert c1 == c2, "Input and output channels must match for residual connection"
        
        hidden_dim = max(c1 // 4, 16)          # 压缩通道，保持轻量
        self.hidden_dim = hidden_dim
        self.use_consistency = use_consistency
        self.lambda_fg = lambda_fg
        self.lambda_consis = lambda_consis

        # 1. 共享降维 Stem
        self.stem = Conv(c1, hidden_dim, k=1)

        self.channel_expand = nn.Conv2d(self.hidden_dim, c1, 1)

        # 2. 前景概率预测分支
        self.foreground_branch = nn.Sequential(
            Conv(hidden_dim, hidden_dim, k=3, g=hidden_dim),  # Depthwise conv
            Conv(hidden_dim, hidden_dim, k=1),
            nn.Conv2d(hidden_dim, 1, kernel_size=1)           # 输出单通道 logits
        )

        # 3. 前景引导的通道注意力分支
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),                # (B, hidden_dim, 1, 1)
            nn.Conv2d(hidden_dim, hidden_dim, 1),    # 无BN/激活，保持原始响应
            nn.Sigmoid()                              # 生成通道权重
        )

        # 4. 可选：用于恢复通道数的投影层（若c1 != hidden_dim，此处直接对x操作，故不需）
        #    因为我们是在原始特征x上直接调制，x的通道数仍为c1，无需额外投影。

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: input feature map, shape (B, C, H, W)
            mask: optional ground-truth foreground mask (B, H, W) or (B, 1, H, W),
                  used only during training.
        Returns:
            out: modulated feature map, same shape as x
            (if training and mask provided, also returns total loss)
        """
        identity = x  # 残差连接用
        B, C, H, W = x.shape

        # --- Stem降维 ---
        feat = self.stem(x)                     # (B, hidden_dim, H, W)

        # --- 前景概率预测 ---
        fg_logits = self.foreground_branch(feat)  # (B, 1, H, W)
        fg_prob = torch.sigmoid(fg_logits)        # 前景概率图 [0,1]

        # --- 前景引导的通道注意力 ---
        # 利用前景概率对特征进行加权池化：只关注前景区域
        # 先将feat与fg_prob相乘，得到前景特征
        fg_feat = feat * fg_prob                  # (B, hidden_dim, H, W)
        # 全局平均池化，得到前景区域的特征描述
        fg_global = fg_feat.mean(dim=(2,3), keepdim=True)  # (B, hidden_dim, 1, 1)
        # 生成通道权重
        channel_weight = self.channel_attn(fg_global)       # (B, hidden_dim, 1, 1)

        # --- 双维度调制 ---
        # 通道权重上采样到与fg_prob相同的空间尺寸（广播即可）
        # 调制项 = x * (channel_weight * fg_prob)  注意channel_weight需投影回C通道
        # 但channel_weight是hidden_dim维，需要先映射回C维，或直接利用原始x？
        # 简单有效的方式：用1x1卷积将channel_weight扩展到C通道，再与x相乘。
        # 为保持轻量，这里直接在hidden_dim空间生成调制信号，然后通过一个可学习的1x1卷积
        # 恢复通道数并加到x上。
        modulation = self.channel_expand(channel_weight * fg_prob)      # (B, C, H, W)
        out = identity + identity * modulation                          # 残差形式

        # --- 损失计算（训练模式）---
        if self.training and mask is not None:
            mask = self._prepare_mask(mask, (H, W))
            loss_fg = self.dice_loss(fg_prob, mask) + self.focal_loss(fg_logits, mask)
            total_loss = self.lambda_fg * loss_fg

            if self.use_consistency:
                loss_consis = self.consistency_loss(feat, fg_prob, mask)
                total_loss = total_loss + self.lambda_consis * loss_consis

            return out, total_loss
        else:
            return out

    # -------------------- 损失函数 --------------------
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice loss for binary foreground prediction."""
        smooth = 1e-6
        pred = pred.contiguous().view(pred.size(0), -1)
        target = target.contiguous().view(target.size(0), -1)
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)
        loss = 1 - (2. * intersection + smooth) / (union + smooth)
        return loss.mean()

    def focal_loss(self, logits: torch.Tensor, target: torch.Tensor, 
                   alpha: float = 0.75, gamma: float = 2.0) -> torch.Tensor:
        """Focal loss for binary classification."""
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        prob = torch.sigmoid(logits)
        pt = torch.where(target == 1, prob, 1 - prob)
        loss = alpha * (1 - pt) ** gamma * bce
        return loss.mean()

    def consistency_loss(self, feat: torch.Tensor, fg_prob: torch.Tensor, 
                         target: torch.Tensor) -> torch.Tensor:
        """
        Foreground consistency loss: encourage features inside foreground region
        to be close to their mean, promoting intra-class compactness.
        """
        # 使用gt mask确定前景区域（也可用fg_prob > 0.5，但gt更准确）
        fg_mask = (target > 0.5).float()               # (B,1,H,W)
        # 前景区域的特征均值
        fg_feat = feat * fg_mask                        # (B, hidden_dim, H, W)
        sum_fg = fg_feat.sum(dim=(2,3), keepdim=True)   # (B, hidden_dim, 1, 1)
        count_fg = fg_mask.sum(dim=(2,3), keepdim=True) + 1e-6
        fg_mean = sum_fg / count_fg                      # 前景中心向量
        # 计算前景像素特征与中心向量的L2距离
        diff = (feat - fg_mean) ** 2
        loss = (diff * fg_mask).sum() / count_fg.sum()
        return loss

    def _prepare_mask(self, masks: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """Resize and normalize mask to target spatial size."""
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks_resized = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)
        return torch.clamp(masks_resized, 0, 1)