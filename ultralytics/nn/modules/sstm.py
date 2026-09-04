"""
自监督调优模块 (Self-Supervised Tuning Module)
基于预生成的二值掩码进行特征调优
"""

import re

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

# class SSTM(nn.Module):
#     """
#     自监督调优模块
    
#     原理:
#     1. 通过轻量分割头预测掩码
#     2. 计算自监督损失（掩码预测）
#     3. 使用梯度引导调整特征：F' = F - α * ∂L/∂F
#     4. 调整后的特征更具判别性
#     """

#     def __init__(self,
#                  c1: int,
#                  c2: int,
#                  alpha: float=0.1,          # 梯度调整系数
#                  lambda_self: float=0.2,    # 自监督损失权重
#                  lambda_edge: float=0.2,    # 边缘损失权重    
#                 ):
#         super().__init__()

#         self.in_channels = c1
#         self._ = c2
#         self.alpha_init = alpha
#         self.lambda_self = lambda_self
#         self.lambda_edge = lambda_edge

#         # 轻量化分割头
#         self.seg_head = nn.Sequential(
#             nn.Conv2d(c1, c1 // 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(c1 // 2),
#             nn.SiLU(),
#             nn.Conv2d(c1 // 2, c1 // 4, kernel_size=3, padding=1),
#             nn.BatchNorm2d(c1 // 4),
#             nn.SiLU(),
#             nn.Conv2d(c1 // 4, 1, kernel_size=1)
#         )

#         # 梯度调整层
#         self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))

#         # 特征增强层
#         self.feature_enhance = nn.Sequential(
#             nn.Conv2d(self.in_channels, self.in_channels, 1),
#             nn.BatchNorm2d(self.in_channels),
#             nn.SiLU()
#         )

#         # 初始化权重
#         self._initialize_weights()

#         print(f"SSTM模块初始化: in_channels={self.in_channels}, alpha={alpha}, "
#               f"lambda_self={lambda_self}, lambda_edge={lambda_edge}")
    
#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
    
#     def forward(self, 
#                 features: torch.Tensor, 
#                 masks: Optional[torch.Tensor]=None,
#                 ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
#         """前向传播

#         Args:
#             features (torch.Tensor): 输入特征 [B, C, H, W]
#             masks (Optional[torch.Tensor], optional): 二值掩码 [B, 1, H, W]. Defaults to None.

#         Returns:
#             enhanced_features: 增强后的特征
#             self_loss: 自监督损失
#         """
#         if self.training and masks is not None:
#             # 训练模式：执行自监督优化
#             return self._train_forward(features, masks)
#         else:
#             # 推理模式：直接返回输入特征
#             features = self.feature_enhance(features)
#             return features
    
#     def _train_forward(self, 
#                        features: torch.Tensor, 
#                        masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         训练模式下的前向传播

#         Args:
#             features (torch.Tensor): 输入特征 [B, C, H, W]
#             masks (torch.Tensor): 二值掩码 [B, 1, H, W]
        
#         Returns:
#             enhanced_features (torch.Tensor): 增强后的特征
#             self_loss (torch.Tensor): 自监督辅助损失
#         """
#         B, C, H, W = features.shape

#         # 1. 调整mask尺寸
#         gt_masks = self._prepare_mask(masks, features.shape[2:])

#         # 2. 自监督任务: 预测伪掩码（注意：这里不应用sigmoid，以用于binary_cross_entropy_with_logits）
#         pred_masks_logits = self.seg_head(features)
#         pred_masks = torch.sigmoid(pred_masks_logits)

#         # 3. 计算自监督损失（传递logits而不是sigmoid后的值）
#         self_loss = self._compute_self_supervised_loss(pred_masks_logits, gt_masks)

#         # 4. 梯度引导特征调整
#         if self_loss.requires_grad:
#             # 确保特征需要梯度
#             features = features.requires_grad_(True)

#             # 计算梯度
#             grad_output = torch.autograd.grad(
#                 outputs=self_loss, 
#                 inputs=features,
#                 retain_graph=True,
#                 create_graph=True,
#                 only_inputs=True
#                 )[0]

#             # 应用梯度
#             alpha = torch.clamp(self.alpha, 0.01, 0.5)
#             enhanced_features = features - alpha * grad_output.detach()

#             # 保证特征范围合理
#             enhanced_features = torch.clamp(enhanced_features, -10, 10)
        
#         else:
#             enhanced_features = features

#         # 5. 特征增强
#         enhanced_features = self.feature_enhance(enhanced_features)
        
#         return enhanced_features, self_loss
    
#     def _prepare_mask(self, masks: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
#         """
#         准备自监督分割的伪掩码

#         Args:
#             masks (torch.Tensor): 二值掩码 [B, 1, H, W]
#             target_size (Tuple[int, int]): 目标特征图尺寸 [H_f, W_f]
        
#         Returns:
#             gt_masks (torch.Tensor): 伪掩码 [B, 1, H_feat, W_feat]
#         """
#         B, C, H, W = masks.shape if masks.dim() == 4 else (masks.shape[0], 1, masks.shape[1], masks.shape[2])

#         if masks.dim() == 3:
#             masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
#         # 调整掩码尺寸到目标特征图尺寸
#         masks_resized = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)

#         masks_resized = torch.clamp(masks_resized, 0, 1)
            
#         return masks_resized

#     def _compute_self_supervised_loss(self, 
#                                       pred_masks: torch.Tensor, 
#                                       gt_masks: torch.Tensor) -> torch.Tensor:
#         """计算自监督分割的辅助损失

#         Args:
#             pred_masks (torch.Tensor): 预测掩码 [B, 1, H, W]
#             gt_masks (torch.Tensor): 伪掩码 [B, 1, H, W]
        
#         Returns:
#             aux_loss (torch.Tensor): 辅助损失
#         """
#         """
#         计算自监督损失
        
#         包含：
#         1. 二元交叉熵损失（主损失）
#         2. 边缘正则化损失（鼓励平滑）
#         """
#         B, _, H, W = pred_masks.shape
        
#         # 1. 二元交叉熵损失（使用binary_cross_entropy_with_logits，这在autocast中是安全的）
#         # 使用pos_weight来平衡正负样本
#         pos_weight = torch.tensor(2.0, device=pred_masks.device)
#         bce_loss = F.binary_cross_entropy_with_logits(
#             pred_masks,
#             gt_masks,
#             reduction='mean',
#             pos_weight=pos_weight
#         )
        
#         # 2. 边缘正则化损失（鼓励预测掩码平滑）
#         # 需要对logits应用sigmoid以计算边缘损失
#         pred_masks_sigmoid = torch.sigmoid(pred_masks)
#         edge_loss = self._compute_edge_regularization(pred_masks_sigmoid)
        
#         # 3. 总自监督损失
#         total_loss = bce_loss + self.lambda_edge * edge_loss
        
#         # 记录损失分量（用于调试）
#         self._loss_components = {
#             'bce_loss': bce_loss.item() if hasattr(bce_loss, 'item') else bce_loss,
#             'edge_loss': edge_loss.item() if hasattr(edge_loss, 'item') else edge_loss
#         }
        
#         return total_loss

#     def _compute_edge_regularization(self, mask: torch.Tensor) -> torch.Tensor:
#         """
#         计算边缘正则化损失
        
#         目的：鼓励预测的掩码边缘平滑，避免过于破碎
#         """
#         # 计算梯度（使用Sobel算子）
#         sobel_x = torch.tensor([[-1, 0, 1], 
#                                [-2, 0, 2], 
#                                [-1, 0, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
#         sobel_y = torch.tensor([[-1, -2, -1], 
#                                [0, 0, 0], 
#                                [1, 2, 1]], dtype=torch.float32, device=mask.device).view(1, 1, 3, 3)
        
#         # 计算梯度
#         grad_x = F.conv2d(mask, sobel_x, padding=1)
#         grad_y = F.conv2d(mask, sobel_y, padding=1)
        
#         # 梯度幅度
#         grad_mag = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8)
        
#         # 边缘损失：鼓励低梯度（平滑）
#         edge_loss = torch.mean(grad_mag)
        
#         return edge_loss
    
#     def get_loss_components(self):
#         """获取损失分量（用于监控）"""
#         return getattr(self, '_loss_components', {})


# class SSTM(nn.Module):
#     """
#     梯度引导特征增强模块（方案二）
#     训练时用真实梯度监督梯度预测网络，推理时直接用预测梯度调整特征
#     """
#     def __init__(self, 
#                  c1: int,
#                  c2: int,
#                  alpha: float = 0.1,
#                  lambda_self: float = 0.2,
#                  lambda_grad: float = 0.1,
#                  lambda_edge: float = 0.2,
#                  learn_alpha: bool = False):
#         super().__init__()
#         self.in_channels = c1
#         self.lambda_self = lambda_self
#         self.lambda_grad = lambda_grad
#         self.lambda_edge = lambda_edge

#         # # 可学习的调整系数（可选）
#         # if learn_alpha:
#         #     self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))
#         # else:
#         #     self.register_buffer('alpha', torch.tensor(alpha, dtype=torch.float32))

#         # # 1. 自监督分割头（用于生成真实梯度）
#         # mid_ch = max(4, self.in_channels // 8)  # 至少4通道
#         # self.seg_head = nn.Sequential(
#         #     # 深度可分离卷积代替标准3x3
#         #     nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, padding=1, groups=self.in_channels),
#         #     nn.Conv2d(self.in_channels, mid_ch, kernel_size=1),  # pointwise降维
#         #     nn.BatchNorm2d(mid_ch),
#         #     nn.SiLU(),
#         #     nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch),
#         #     nn.Conv2d(mid_ch, 1, kernel_size=1)
#         # )

#         # # 2. 梯度预测网络（输入特征，输出与特征同尺寸的“梯度图”）
#         # self.grad_predictor = nn.Sequential(
#         #     # 先降维
#         #     nn.Conv2d(self.in_channels, mid_ch, kernel_size=1),
#         #     nn.BatchNorm2d(mid_ch),
#         #     nn.SiLU(),
#         #     # 深度可分离3x3
#         #     nn.Conv2d(mid_ch, mid_ch, kernel_size=3, padding=1, groups=mid_ch),
#         #     nn.Conv2d(mid_ch, mid_ch, kernel_size=1),
#         #     nn.BatchNorm2d(mid_ch),
#         #     nn.SiLU(),
#         #     # 升维回 in_channels
#         #     nn.Conv2d(mid_ch, self.in_channels, kernel_size=1)
#         # )

#         # # 3. 特征增强层（可选，用于进一步提升特征）
#         # self.feature_enhance = nn.Sequential(
#         #     nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1),
#         #     nn.BatchNorm2d(self.in_channels),
#         #     nn.SiLU()
#         # )

#         # self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)

#     def _prepare_mask(self, masks: torch.Tensor, target_size):
#         """将输入掩码调整到特征图尺寸"""
#         if masks.dim() == 3:
#             masks = masks.unsqueeze(1)
#         masks_resized = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)
#         return torch.clamp(masks_resized, 0, 1)

#     def _compute_self_loss(self, pred_logits, gt_masks):
#         """自监督损失：BCE + 边缘平滑"""
#         # 二元交叉熵（使用logits）
#         pos_weight = torch.tensor(2.0, device=pred_logits.device)
#         bce = F.binary_cross_entropy_with_logits(pred_logits, gt_masks, pos_weight=pos_weight)

#         # 边缘平滑损失（对sigmoid后的mask计算梯度幅值）
#         pred_masks = torch.sigmoid(pred_logits)
#         sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=pred_logits.device).view(1,1,3,3)
#         sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=pred_logits.device).view(1,1,3,3)
#         grad_x = F.conv2d(pred_masks, sobel_x, padding=1)
#         grad_y = F.conv2d(pred_masks, sobel_y, padding=1)
#         grad_mag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
#         edge_loss = torch.mean(grad_mag)

#         return bce + self.lambda_edge * edge_loss

#     def forward(self, x, masks=None):
#         """
#         Args:
#             x: 输入特征 [B, C, H, W]
#             masks: 二值掩码 [B, 1, H, W] 或 [B, H, W]（仅训练时需要）
#         Returns:
#             enhanced_feat: 增强后的特征 [B, C, H, W]
#             loss_self: 自监督损失（标量Tensor或None）
#             loss_grad: 梯度匹配损失（标量Tensor或None）
#         """
#         if self.training and masks is not None:
#             # ---------- 训练模式 ----------
#             # 1. 自监督损失（必须保留计算图，用于后续梯度计算）
#             pred_logits = self.seg_head(x)
#             gt_masks = self._prepare_mask(masks, x.shape[2:])
#             loss_self = self._compute_self_loss(pred_logits, gt_masks)

#             # 2. 计算真实梯度（对输入特征x的梯度）
#             # 设置create_graph=True以便后续计算二阶梯度（如果需可学习alpha）
#             grad_real = torch.autograd.grad(
#                 loss_self, x, retain_graph=True, create_graph=True
#             )[0]

#             # 3. 预测梯度
#             grad_pred = self.grad_predictor(x)

#             # 4. 梯度匹配损失（监督grad_predictor）
#             loss_grad = F.mse_loss(grad_pred, grad_real.detach())  # detach真实梯度，避免梯度回传到分割头

#             # 5. 特征调整
#             alpha = torch.clamp(self.alpha, 0.01, 0.5) if isinstance(self.alpha, torch.Tensor) else self.alpha
#             x_adj = x - alpha * grad_pred

#             # 6. 特征增强
#             # enhanced = self.feature_enhance(x_adj)

#             return x_adj, loss_self, loss_grad

#         else:
#             # ---------- 推理模式 ----------
#             grad_pred = self.grad_predictor(x)
#             alpha = torch.clamp(self.alpha, 0.01, 0.5) if isinstance(self.alpha, torch.Tensor) else self.alpha
#             x_adj = x - alpha * grad_pred
#             # enhanced = self.feature_enhance(x_adj)
#             return x_adj


# class SSTM(nn.Module):
#     """
#     梯度引导特征增强模块（方案二）
#     训练时用真实梯度监督梯度预测网络，推理时直接用预测梯度调整特征
#     """
#     def __init__(self, 
#                  c1: int,
#                  c2: int,
#                  alpha: float = 0.1,
#                  lambda_self: float = 0.2,
#                  lambda_grad: float = 0.1,
#                  lambda_edge: float = 0.2,
#                  learn_alpha: bool = False):
#         super().__init__()
#         self.in_channels = c1
#         self.lambda_self = lambda_self
#         self.lambda_grad = lambda_grad
#         self.lambda_edge = lambda_edge

#     def forward(
#         self,
#         feat: torch.Tensor,       # 特征图 [B, C, H_feat, W_feat]
#         masks: torch.Tensor = None   # 原图尺寸真值图 [B, 1, H_img, W_img]
#     ) -> torch.Tensor:
#         """
#         无特征降维的损失计算：
#         1. 将GT下采样到特征层尺寸
#         2. 广播到特征图相同通道数
#         3. 计算多通道MSE，完整保留所有通道信息
#         """
#         B, C, Hf, Wf = feat.shape
        
#         if self.training and masks is not None:
#             # 步骤1：将GT下采样到特征层尺寸
#             gt_feat = F.interpolate(
#                 masks,
#                 size=(Hf, Wf),
#                 mode='bilinear',
#                 align_corners=False
#             )  # [B, 1, Hf, Wf]
            
#             # 步骤2：广播到特征图相同通道数（不压缩特征，保留所有信息）
#             gt_broadcast = gt_feat.expand(B, C, -1, -1)  # [B, C, Hf, Wf]
            
#             # 步骤3：计算多通道MSE损失
#             loss_scale = ((feat - gt_feat) ** 2).mean()
#         else:
#             return feat
        
#         return feat, loss_scale


class SSTM(nn.Module):
    """
    基于逐点正交约束的互补特征学习模块。
    输入：主干特征 x (B, C, H, W)
    输出：融合后的增强特征 z (B, C, H, W) 以及可选的组合损失。
    模块内部计算正交损失、稀疏损失，并可选的 mask 监督损失，需与外部检测损失相加构成总损失。
    """
    def __init__(self, c1: int, c2: int, hidden_channels=None, out_channels=None,
                 kernel_size=3, lambda_orth=20.0, lambda_sparse=2.0, lambda_mask=1.0, fuse_mode='concat', eps=1e-12):
        """
        Args:
            c1: 输入特征通道数（即主干特征的通道数）
            c2: 输出特征通道数
            hidden_channels: 互补生成网络隐藏层通道数，默认与 c1 相同
            out_channels: 互补特征输出通道数，默认与 in_channels 相同（用于相加融合）
            kernel_size: 卷积核大小，默认 3
            lambda_orth: 正交损失系数
            lambda_sparse: 稀疏损失系数
            lambda_mask: mask 监督损失系数
            eps: 归一化时防止除零的小量
        """
        super().__init__()
        self.lambda_orth = lambda_orth
        self.lambda_sparse = lambda_sparse
        self.lambda_mask = lambda_mask
        self.eps = eps

        hidden = hidden_channels if hidden_channels else c1
        out = c1  # 保持输出通道与输入一致，方便融合

        # 互补特征生成网络（简单的卷积堆叠，保持空间尺寸）
        self.complement_net = nn.Sequential(
            # Depthwise 卷积
            nn.Conv2d(c1, c1, kernel_size, padding=kernel_size//2, groups=c1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU(inplace=True),
            # 通道压缩
            nn.Conv2d(c1, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            # 通道恢复
            nn.Conv2d(hidden, c1, 1, bias=True)
        )

        # 融合层（可学习加权）
        self.fusion_weight = nn.Parameter(torch.ones(1) * 0.5)

    def forward(self, x: torch.Tensor, masks: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：生成互补特征并融合，同时计算损失。
        Args:
            x: 主干特征 (B, C, H, W)
            mask: 可选的 mask 用于监督注意力图 (B, H, W) 或 (B, 1, H, W)，值域 [0,1]
        Returns:
            z: 融合后的增强特征 (B, C, H, W)
            total_loss: 标量张量，组合损失（若 mask 为 None 则不包含 mask 损失）
        """
        y = self.complement_net(x)
        z = x + self.fusion_weight * y
        if self.training:
            total_loss = self.compute_losses(x, y, masks)
            return z, total_loss
        else:
            return z

    def _prepare_mask(self, masks: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """将输入掩码调整到目标特征图尺寸，并确保为单通道 [0,1] 范围。"""
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)                  # (B,1,H,W)
        masks_resized = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)
        return torch.clamp(masks_resized, 0, 1)
    
    def compute_orth_loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """通道级去相关损失（替代逐点正交）"""
        B, C, H, W = x.shape
        
        # 展平空间维度
        x_flat = x.view(B, C, -1)  # (B, C, H*W)
        y_flat = y.view(B, C, -1)
        
        # 计算通道间协方差矩阵
        x_mean = x_flat.mean(dim=-1, keepdim=True)
        y_mean = y_flat.mean(dim=-1, keepdim=True)
        x_centered = x_flat - x_mean
        y_centered = y_flat - y_mean
        
        # 协方差矩阵 (B, C, C)
        cov = torch.bmm(x_centered, y_centered.transpose(1, 2)) / (H * W)
        
        # 强制协方差接近对角阵（去相关）
        target = torch.eye(C, device=x.device).unsqueeze(0).expand(B, -1, -1)  # (B, C, C)
        orth_loss = F.mse_loss(cov, target)
        
        return orth_loss
    
    def compute_losses(self, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor = None) -> Tuple[torch.Tensor, dict]:
        """
        计算总损失，可选梯度平衡
        """
        # 1. 正交损失（通道级去相关）
        orth_loss = self.compute_orth_loss(x, y)
        
        # 2. 稀疏损失（L1）
        B, C, H, W = y.shape
        if mask is not None:
            mask_prepared = self._prepare_mask(mask, (H, W))  # (B, 1, H, W)
            mask_prepared = mask_prepared.expand(B, C, -1, -1)  # (B, C, H, W)
        sparse_loss = torch.mean(torch.abs(y) * mask_prepared)
        
        if self.training:
            total_loss = (self.lambda_orth * orth_loss +
                            self.lambda_sparse * sparse_loss)
            
            losses_dict = {
                'orth_loss': orth_loss.item(),
                'sparse_loss': sparse_loss.item(),
            }
        
        return total_loss


if __name__ == "__main__":
    """测试SSTM模块的功能"""
    print("=== SSTM模块测试 ===\n")
    
    # 1. 设置基本参数
    batch_size = 2
    in_channels = 256
    feat_h, feat_w = 40, 40  # 特征图尺寸
    mask_h, mask_w = 640, 640  # 掩码尺寸
    
    # 2. 创建测试数据
    print("1. 创建测试数据...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   使用设备: {device}")
    
    # 输入特征
    features = torch.randn(batch_size, in_channels, feat_h, feat_w, device=device)
    features.requires_grad_(True)  # 确保特征需要梯度
    print(f"   特征图尺寸: {features.shape}")
    
    # 二值掩码（模拟分割任务的GT掩码）
    masks = torch.randn(batch_size, 1, mask_h, mask_w, device=device)
    masks = torch.sigmoid(masks)  # 转换为[0,1]范围
    print(f"   掩码尺寸: {masks.shape}")
    print(f"   掩码值范围: [{masks.min().item():.3f}, {masks.max().item():.3f}]")
    
    # 3. 创建SSTM模块
    print("\n2. 创建SSTM模块...")
    sstm = SSTM(
        c1=in_channels,
        c2=128,
        alpha=0.1,
        lambda_self=0.2,
        lambda_edge=0.2
    ).to(device)
    
    # 4. 测试训练模式
    print("\n3. 测试训练模式...")
    sstm.train()
    
    # 前向传播（带掩码）
    enhanced_features, self_loss = sstm(features, masks)
    
    print(f"   原始特征形状: {features.shape}")
    print(f"   增强特征形状: {enhanced_features.shape}")
    print(f"   自监督损失值: {self_loss.item():.6f}")
    
    # 检查特征是否被修改
    if torch.allclose(enhanced_features, features, rtol=1e-5):
        print("   ⚠️ 警告: 增强特征与原始特征几乎相同，梯度调整可能未生效")
    else:
        print("   ✓ 特征已被成功调整")
    
    # 检查特征值范围
    min_val, max_val = enhanced_features.min().item(), enhanced_features.max().item()
    print(f"   增强特征值范围: [{min_val:.4f}, {max_val:.4f}]")
    
    # 5. 测试推理模式
    print("\n4. 测试推理模式...")
    sstm.eval()
    
    # 推理模式（无掩码）
    infer_features, infer_loss = sstm(features, masks=None)
    
    print(f"   推理输出特征形状: {infer_features.shape}")
    print(f"   推理损失值: {infer_loss}")
    
    # 推理模式应该只进行特征增强，不计算损失
    if infer_loss is None:
        print("   ✓ 推理模式正确返回None损失")
    else:
        print("   ❌ 推理模式返回了损失值，可能存在错误")
    
    # 6. 梯度测试
    print("\n5. 测试梯度传播...")
    sstm.train()
    
    # 前向传播
    enhanced_features, self_loss = sstm(features, masks)
    
    # 计算总损失并反向传播
    total_loss = self_loss * sstm.lambda_self
    total_loss.backward()
    
    # 检查梯度
    has_gradients = features.grad is not None and not torch.all(features.grad == 0)
    
    if has_gradients:
        print(f"   ✓ 梯度计算正常")
        print(f"   特征梯度范数: {torch.norm(features.grad):.6f}")
    else:
        print("   ❌ 梯度计算异常，可能未正确传播")
    
    # 7. 测试掩码预处理
    print("\n6. 测试掩码预处理...")
    target_size = (feat_h, feat_w)
    prepared_masks = sstm._prepare_mask(masks, target_size)
    
    print(f"   原始掩码形状: {masks.shape}")
    print(f"   预处理后掩码形状: {prepared_masks.shape}")
    print(f"   预处理掩码值范围: [{prepared_masks.min().item():.3f}, {prepared_masks.max().item():.3f}]")
    
    # 检查尺寸是否正确
    if prepared_masks.shape[2:] == target_size:
        print("   ✓ 掩码尺寸调整正确")
    else:
        print("   ❌ 掩码尺寸调整错误")
    
    # 8. 测试损失分量
    print("\n7. 测试损失分量...")
    loss_components = sstm.get_loss_components()
    print(f"   损失分量: {loss_components}")
    
    if loss_components:
        print("   ✓ 成功获取损失分量")
    else:
        print("   ⚠️ 未获取到损失分量，可能在推理模式下")
    
    # 9. 测试不同参数配置
    print("\n8. 测试不同参数配置...")
    
    # 创建不同配置的SSTM模块
    sstm_configs = [
        {"alpha": 0.05, "lambda_self": 0.1, "lambda_edge": 0.1},
        {"alpha": 0.2, "lambda_self": 0.3, "lambda_edge": 0.3},
        {"alpha": 0.01, "lambda_self": 0.5, "lambda_edge": 0.1}
    ]
    
    for i, config in enumerate(sstm_configs):
        sstm_custom = SSTM(
            c1=in_channels,
            c2=128,
            alpha=config["alpha"],
            lambda_self=config["lambda_self"],
            lambda_edge=config["lambda_edge"]
        ).to(device)
        
        sstm_custom.train()
        _, loss = sstm_custom(features, masks)
        
        print(f"   配置{i+1}: alpha={config['alpha']}, lambda_self={config['lambda_self']}, "
              f"lambda_edge={config['lambda_edge']} -> 损失: {loss.item():.6f}")
    
    print("\n=== 测试完成 ===")
    
    # 总结
    print("\n测试总结:")
    print(f"- 训练模式: {'通过' if not torch.allclose(enhanced_features, features) else '警告'}")
    print(f"- 推理模式: {'通过' if infer_loss is None else '失败'}")
    print(f"- 梯度传播: {'通过' if has_gradients else '失败'}")
    print(f"- 掩码处理: {'通过' if prepared_masks.shape[2:] == target_size else '失败'}")
    print(f"- 参数配置: 测试了{len(sstm_configs)}种不同配置")