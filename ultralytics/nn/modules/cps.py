from ultralytics.nn.modules.conv import Conv, autopad
from ultralytics.nn.modules.block import Bottleneck, C2f
from ultralytics.nn.modules.scjw import SCJW

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple, Optional

# ------------------- 外围卷积 (简化版：大核DWConv + 核级位置编码) -------------------
class PeripheralConv(nn.Module):
    """
    外围卷积：大核深度卷积 + 可学习位置编码
    - 使用深度卷积 (groups=输入通道) 保持高效
    - 在卷积核权重上叠加位置编码，使共享区域也能感知相对位置
    """
    def __init__(self, in_ch, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        # 深度卷积权重 [in_ch, 1, k, k]
        self.dwconv = nn.Conv2d(in_ch, in_ch, kernel_size, stride=1,
                                 padding=kernel_size//2, groups=in_ch, bias=False)
        # 核级位置编码：与卷积核相同形状的可学习参数
        self.pos_embed = nn.Parameter(torch.zeros(1, in_ch, kernel_size, kernel_size))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.bn = nn.BatchNorm2d(in_ch)
        self.act = nn.SiLU()

    def forward(self, x):
        # 将位置编码加到卷积核权重上
        effective_weight = self.dwconv.weight + self.pos_embed
        # 使用修改后的权重进行卷积（利用已有的dwconv的其他参数）
        out = F.conv2d(x, effective_weight, stride=1,
                       padding=self.kernel_size//2, groups=x.shape[1])
        return self.act(self.bn(out))

# ------------------- 空间-语义互补单元 (CMU) -------------------
class CMU(nn.Module):
    """
    互补映射单元：在一个分支内实现空间-语义信息互补
    - 按比例α拆分为空间流和语义流
    - 空间流用1x1卷积保持精细结构；语义流用深度卷积（大核时用PeripheralConv）扩大感受野
    - 交叉门控：语义信息指导空间流，空间信息指导语义流
    """
    def __init__(self, c, alpha=0.5, kernel_size=3):
        super().__init__()
        self.alpha = alpha
        c_spa = int(c * alpha)
        c_sem = c - c_spa

        self.spa_conv = Conv(c_spa, c_spa, 1)  # 保留空间细节
        # 语义分支：若核大小>7则用外围卷积，否则用普通深度卷积
        if kernel_size > 7:
            self.sem_conv = PeripheralConv(c_sem, kernel_size)
        else:
            self.sem_conv = Conv(c_sem, c_sem, kernel_size, g=c_sem)

        # 门控生成器
        self.gate_spa = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv(c_sem, c_spa, 1, act=False),
            nn.Sigmoid()
        )
        self.gate_sem = nn.Sequential(
            Conv(c_spa, c_sem, 1, act=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        xs, xc = x.split([int(self.alpha * x.shape[1]),
                          x.shape[1] - int(self.alpha * x.shape[1])], dim=1)
        xs = self.spa_conv(xs)
        xc = self.sem_conv(xc)

        # 语义门控 → 加权空间分支
        xs = xs * self.gate_spa(xc)
        # 空间门控 → 加权语义分支
        xc = xc * self.gate_sem(xs)

        return torch.cat([xs, xc], dim=1)

# ------------------- 多原型通道注意力 -------------------
class MultiProtoChannelAttn(nn.Module):
    """
    多原型通道注意力
    - 使用可学习的原型向量集合
    - 根据前景特征与各原型的相似度，融合得到通道权重向量
    """
    def __init__(self, channels, num_protos=4):
        super().__init__()
        self.num_protos = num_protos
        self.channels = channels
        self.prototypes = nn.Parameter(torch.randn(num_protos, channels))
        nn.init.kaiming_uniform_(self.prototypes, a=0.1)

    def forward(self, x, mask):
        """
        x: [B, C, H, W]
        mask: [B, 1, H, W] 前景掩码 (1为前景)
        返回: [B, C, 1, 1] 通道权重
        """
        B, C = x.shape[:2]
        # 计算前景全局特征 (平均)
        fore_feat = (x * mask).sum(dim=[2,3], keepdim=True) / mask.sum(dim=[2,3], keepdim=True).clamp(min=1)
        fore_feat = fore_feat.squeeze(-1).squeeze(-1)  # [B, C]

        # 归一化
        fore_norm = F.normalize(fore_feat, dim=1)
        proto_norm = F.normalize(self.prototypes, dim=1)  # [P, C]

        # 计算每个样本与每个原型的余弦相似度 [B, P]
        sim = fore_norm @ proto_norm.t()

        # 用softmax得到原型权重，加权融合原型
        proto_weight = F.softmax(sim, dim=1)  # [B, P]
        fused_proto = proto_weight @ self.prototypes  # [B, C]

        # 将融合原型作为通道重要性，经sigmoid映射到(0,1)
        channel_weight = torch.sigmoid(fused_proto).view(B, C, 1, 1)
        return channel_weight

# ------------------- 外围空间注意力 -------------------
class PeripheralSpatialAttn(nn.Module):
    """
    外围空间注意力
    - 下采样特征图计算相似度，降低计算量
    - 添加位置编码补偿共享导致的模糊
    """
    def __init__(self, scale_factor=4):
        super().__init__()
        self.scale = scale_factor
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, 1, 1))  # 简化版：可学习标量

    def forward(self, x, proto):
        """
        x: [B, C, H, W]
        proto: [B, C, 1, 1] 前景原型 (通常为前景平均特征)
        返回: [B, 1, H, W] 空间权重
        """
        B, C, H, W = x.shape
        H_low, W_low = H // self.scale, W // self.scale

        # 下采样
        x_low = F.avg_pool2d(x, self.scale)  # [B, C, H_low, W_low]
        x_low_norm = F.normalize(x_low, dim=1)
        proto_norm = F.normalize(proto, dim=1)  # [B, C, 1, 1]

        # 计算低分辨率上的余弦相似度
        sim_low = (x_low_norm * proto_norm).sum(dim=1, keepdim=True)  # [B, 1, H_low, W_low]

        # 添加位置编码（简单加常数）
        sim_low = sim_low + self.pos_embed

        # 上采样回原分辨率
        sim = F.interpolate(sim_low, size=(H, W), mode='bilinear', align_corners=False)
        return torch.sigmoid(sim)
    

class ContrastAttention(nn.Module):
    def __init__(self, in_channels, tau=2.0):
        super().__init__()
        self.tau = tau

    def forward(self, x, mask):
        B, C, H, W = x.shape
        if mask is None:
            return x
        mask = F.interpolate(mask, (H, W), mode='nearest')
        x_flat = x.view(B, C, -1)
        mask_flat = mask.view(B, 1, -1)

        # 计算前景和背景原型
        fore_proto = []
        back_proto = []
        for b in range(B):
            fg = x_flat[b, :, mask_flat[b, 0] > 0]  # [C, M]
            bg = x_flat[b, :, mask_flat[b, 0] == 0] # [C, N]
            # 修复：添加 dtype=x.dtype 确保类型一致
            fore_proto.append(fg.mean(dim=1, keepdim=True) if fg.size(1)>0 else torch.zeros(C,1, dtype=x.dtype).to(x.device))
            back_proto.append(bg.mean(dim=1, keepdim=True) if bg.size(1)>0 else torch.zeros(C,1, dtype=x.dtype).to(x.device))
        p_f = torch.stack(fore_proto, dim=0).squeeze(-1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        p_b = torch.stack(back_proto, dim=0).squeeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 余弦相似度
        x_norm = F.normalize(x, dim=1)
        p_f_norm = F.normalize(p_f, dim=1)
        p_b_norm = F.normalize(p_b, dim=1)
        sim_f = (x_norm * p_f_norm).sum(dim=1, keepdim=True)  # [B,1,H,W]
        sim_b = (x_norm * p_b_norm).sum(dim=1, keepdim=True)

        # 对比权重
        weight = torch.exp(self.tau * sim_f) / (torch.exp(self.tau * sim_f) + torch.exp(self.tau * sim_b) + 1e-6)
        # 确保 weight 与 x 类型一致
        weight = weight.to(dtype=x.dtype)
        return x * weight


# ------------------- CPS-Block 主模块 -------------------
class CPSBlock(nn.Module):
    """
    中央-外围协同感知模块
    可替换YOLOv8主干中的C2f层
    Args:
        c1: 输入通道数
        c2: 输出通道数
        n: 重复次数（本实现中固定为1，如需加深可在外部堆叠）
        alpha: 空间流占比 (0~1)
        kernel_sizes: 两个分支的卷积核大小列表，例如 [3,5] 或 [5,7]
        num_protos: 多原型注意力中原型数量
    """
    def __init__(self, c1, c2, n=1, alpha=0.5, kernel_sizes=[3,5], num_protos=4, use_attn=False):
        super().__init__()
        self.n = n  # 保留接口，实际未使用
        hidden = c2

        # 1x1扩展通道，准备两个分支
        self.conv1 = Conv(c1, hidden * 2, 1)

        # 两个互补映射单元
        self.cmu1 = CMU(hidden, alpha, kernel_sizes[0])
        self.cmu2 = CMU(hidden, alpha, kernel_sizes[1])

        # 融合卷积
        self.conv2 = Conv(hidden * 2, c2, 1)

        # 双域注意力
        self.channel_attn = MultiProtoChannelAttn(c2, num_protos)
        self.spatial_attn = PeripheralSpatialAttn(scale_factor=4)

        self.feature_enhance = ContrastAttention(c2)  # 可选的特征增强模块

        # 残差连接
        self.shortcut = (c1 == c2)
        if not self.shortcut:
            self.cv_shortcut = Conv(c1, c2, 1)

    def _prepare_mask(self, masks: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        """
        准备自监督分割的伪掩码

        Args:
            masks (torch.Tensor): 二值掩码 [B, 1, H, W]
            target_size (Tuple[int, int]): 目标特征图尺寸 [H_f, W_f]
        
        Returns:
            gt_masks (torch.Tensor): 伪掩码 [B, 1, H_feat, W_feat]
        """
        B, C, H, W = masks.shape if masks.dim() == 4 else (masks.shape[0], 1, masks.shape[1], masks.shape[2])

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)  # [B, 1, H, W]
        
        # 调整掩码尺寸到目标特征图尺寸
        masks_resized = F.interpolate(masks, size=target_size, mode='bilinear', align_corners=False)

        masks_resized = torch.clamp(masks_resized, 0, 1)
            
        return masks_resized

    def forward(self, x, mask=None):
        identity = x

        # 扩展并分成两路
        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)

        # 通过CMU
        out1 = self.cmu1(x1)
        out2 = self.cmu2(x2)

        # 拼接并融合
        out = torch.cat([out1, out2], dim=1)
        out = self.conv2(out)

        # 如果提供掩码，应用前景引导的注意力
        if mask is not None:
            # 1. 调整mask尺寸
            # mask = self._prepare_mask(mask, out.shape[2:])
            # # 计算前景原型 (用于空间注意力)
            # proto = (out * mask).sum(dim=[2,3], keepdim=True) / mask.sum(dim=[2,3], keepdim=True).clamp(min=1)
            # r_c = self.channel_attn(out, mask)      # [B, C, 1, 1]
            # r_s = self.spatial_attn(out, proto)     # [B, 1, H, W]
            # out = out * r_c * r_s
            out = self.feature_enhance(out, mask)
        else:
            # 1. 调整mask尺寸
            # mask = self._prepare_mask(mask, out.shape[2:])
            # # 计算前景原型 (用于空间注意力)
            # proto = (out * mask).sum(dim=[2,3], keepdim=True) / mask.sum(dim=[2,3], keepdim=True).clamp(min=1)
            # r_c = self.channel_attn(out, mask)      # [B, C, 1, 1]
            # r_s = self.spatial_attn(out, proto)     # [B, 1, H, W]
            # out = out * r_c * r_s
            B, _, H, W = out.shape
            # 创建值为 1/(H*W) 的全图均值掩码
            mean_mask = torch.ones(B, 1, H, W, device=out.device, dtype=out.dtype) / (H * W)
            out = self.feature_enhance(out, mean_mask)

        # 残差连接
        if self.shortcut:
            out = out + identity
        else:
            out = out + self.cv_shortcut(identity)

        return out
    

# class Teacher(nn.Module):
#     """Faster Implementation of CSP Bottleneck with 2 convolutions."""

#     def __init__(self, c1: int, c2: int, n: int = 1, shortcut: bool = False, g: int = 1, e: float = 0.5):
#         """
#         Initialize a CSP bottleneck with 2 convolutions.

#         Args:
#             c1 (int): Input channels.
#             c2 (int): Output channels.
#             n (int): Number of Bottleneck blocks.
#             shortcut (bool): Whether to use shortcut connections.
#             g (int): Groups for convolutions.
#             e (float): Expansion ratio.
#         """
#         super().__init__()
#         self.c = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, 2 * self.c, 1, 1)
#         self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
#         self.cv3 = Conv(c2, c2, 1)
#         self.gamma = nn.Parameter(torch.tensor(0.0))  # 可学习缩放
#         self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
#         self.feature_enhance = ContrastAttention(c2)  # 可选的特征增强模块

#     def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
#         """Forward pass through C2f layer."""
#         y = list(self.cv1(x).chunk(2, 1))
#         y.extend(m(y[-1]) for m in self.m)
#         out = self.cv2(torch.cat(y, 1))
#         out = self.feature_enhance(out, mask)  # 可选的特征增强 + 残差连接 残差连接

#         return out


# class CPSBlock(nn.Module):
#     def __init__(self, c1, c2, n=1, shortcut=False, use_attn=False, g=1, e=0.5):
#         super().__init__()
#         self.teacher = Teacher(c1, c2, n, shortcut, g, e)
#         self.student = C2f(c1, c2, n, shortcut, g, e)
#         self.distill_weight = 50  # 蒸馏损失的权重
#         self.temperature = 4.0  # softmax 温度，用于软化分布

#     def forward(self, x, mask=None):
#         if self.training:
#             # 训练模式：必须提供 mask
#             if mask is None:
#                 raise ValueError("CPSBlock requires mask in training mode")
#             # 教师前向（需要 mask）
#             teacher_out = self.teacher(x, mask)
#             # 学生前向（不需要 mask）
#             student_out = self.student(x)

#             # 计算蒸馏损失（KL散度）
#             # 将特征图视为分布：对每个位置在通道维做 softmax
#             # 先除以温度软化
#             student_logits = student_out / self.temperature
#             teacher_logits = teacher_out.detach() / self.temperature  # 教师不参与梯度

#             # 对每个空间位置计算 KL 散度，再取平均
#             B, C, H, W = student_out.shape
#             student_prob = F.log_softmax(student_logits, dim=1)  # [B, C, H, W]
#             teacher_prob = F.softmax(teacher_logits, dim=1)      # [B, C, H, W]
#             kl_div = F.kl_div(student_prob, teacher_prob, reduction='batchmean')  # 已对 batch 和空间平均

#             # 也可选择 MSE 作为替代，但用户指定 KL
#             # distill_loss = kl_div * self.temperature * self.temperature  # 乘以 temperature^2 以补偿缩放
#             distill_loss = F.mse_loss(student_out, teacher_out.detach())  # 乘以 temperature^2 以补偿缩放

#             return teacher_out, self.distill_weight * distill_loss
#         else:
#             # 推理模式：直接使用学生分支（无需 mask）
#             return self.student(x)