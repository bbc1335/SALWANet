import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3k2, C2f, Bottleneck, Focus, ChannelAttention

from einops import rearrange
import math


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


# 复用之前的RMSNorm2D
class RMSNorm2D(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(dim=1, keepdim=True).sqrt() + self.eps
        x = x / norm * self.weight.view(1, -1, 1, 1)
        return x

# 核心：Stage全局特征表示生成器
class StageGlobalRepr(nn.Module):
    """
    为每个Stage生成全局特征表示，用于跨Stage注意力的Key/Value
    输入：Stage输出的特征图 (B, C_in, H, W)
    输出：1. 全局表示向量 (B, C_out)；2. 对齐后的空间特征图 (B, C_out, H, W)
    """
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c_out = c_out
        # 通道对齐卷积：统一所有Stage的通道数，解决通道不匹配问题
        self.channel_align = Conv(c_in, c_out, k=1, s=1, act=False)
        # 全局上下文编码：轻量级通道注意力，强化全局表示的判别性
        self.global_encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_out, c_out // 4, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(c_out // 4, c_out, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 1. 通道对齐
        x_aligned = self.channel_align(x)  # (B, C_out, H, W)
        # 2. 全局上下文加权
        global_weight = self.global_encoder(x_aligned)  # (B, C_out, 1, 1)
        x_weighted = x_aligned * global_weight
        # 3. 生成全局表示向量（Key/Value用）
        global_repr = x_weighted.mean(dim=(2, 3))  # (B, C_out)
        # 4. 返回全局向量 + 对齐后的空间特征图（Neck融合用）
        return global_repr, x_aligned
    

class CS_AttnRes_Neck(nn.Module):
    """
    修改后的Neck跨Stage注意力残差模块
    核心修改：
    1. 重新设计初始化方式，避免全零问题
    2. 添加c_target_reduced属性，控制目标特征通道降维
    3. 完全删除高斯图相关代码
    """
    def __init__(self, c1, c2, num_heads=1, c_target_reduced=False, use_class_freq=False, class_freq=None, lambda_reg=0.1):
        """
        Args:
            c1: [目标特征通道数, 当前特征通道数]
            c2: 输出通道数（应与当前特征通道数相同，以保证残差连接）
            num_heads: 注意力头数
            use_class_freq: 是否使用类别频率感知
            class_freq: 类别频率列表
            lambda_reg: 类别感知强度
            c_target_reduced: 是否对目标特征通道降维（若True，降维至当前特征通道数）
        """
        super().__init__()
        self.num_heads = num_heads
        self.c_target = c1[0]   # 目标特征原始通道数
        self.c_current = c1[1]  # 当前特征通道数
        self.head_dim = self.c_current // num_heads
        assert self.head_dim * num_heads == self.c_current, "当前特征通道数必须能被头数整除"

        # --------------------------
        # 1. 目标特征通道降维控制
        # --------------------------
        self.c_target_reduced = c_target_reduced
        if self.c_target_reduced:
            # 降维：目标特征 → 当前特征通道数
            self.target_downsample = Conv(self.c_target, self.c_current, k=1, s=1, act=False)
            target_in_channels = self.c_current   # 降维后输入kv_proj的通道数
        else:
            self.target_downsample = nn.Identity()
            target_in_channels = self.c_target     # 保持原始通道数

        # --------------------------
        # 2. 类别频率感知参数
        # --------------------------
        self.use_class_freq = use_class_freq
        if self.use_class_freq:
            self.class_freq = class_freq if class_freq is not None else [1.0] * 10
            self.lambda_reg = lambda_reg
            # 预计算对数逆频率，作为正则项系数
            inv_freq = [1.0 / (f + 1e-8) for f in self.class_freq]
            log_inv_freq = torch.tensor([math.log(f) for f in inv_freq], dtype=torch.float32)
            self.register_buffer("log_inv_freq", log_inv_freq)

        # --------------------------
        # 3. 投影层
        # --------------------------
        self.q_proj = Conv(self.c_current, self.c_current, k=1, s=1, act=False)   # Query投影
        self.kv_proj = Conv(target_in_channels, self.c_current, k=1, s=1, act=False)  # Key/Value投影
        self.out_proj = Conv(self.c_current, c2, k=1, s=1, act=False)              # 输出投影

        # --------------------------
        # 4. 初始化（避免全零问题）
        # --------------------------
        # q_proj使用小方差正态初始化，保证初始时输出变化小但不为零
        nn.init.normal_(self.q_proj.conv.weight, mean=0.0, std=0.01)
        if self.q_proj.conv.bias is not None:
            nn.init.zeros_(self.q_proj.conv.bias)
        # kv_proj和out_proj使用默认的Kaiming初始化（Conv默认）
        # 归一化层
        self.norm_q = nn.LayerNorm(self.c_current)
        self.norm_k = nn.LayerNorm(self.c_current)

    def forward(self, feat):
        """
        Args:
            feat: (target_feat, current_feat)
                target_feat: 目标Stage特征 (B, c_target, H_t, W_t)
                current_feat: 当前Neck层特征 (B, c_current, H, W)
        Returns:
            融合后的特征 (B, c2, H, W)，要求c2 == c_current以保证残差相加
        """
        target_feat, current_feat = feat[0], feat[1]  # 解包
        B, C, H, W = current_feat.shape
        N = H * W

        # --------------------------
        # Step 1: 目标特征降维（若启用）
        # --------------------------
        target_feat = self.target_downsample(target_feat)  # (B, target_in_channels, H_t, W_t)

        # --------------------------
        # Step 2: 分辨率对齐 + Key/Value投影
        # --------------------------
        target_aligned = F.interpolate(target_feat, size=(H, W), mode='bilinear', align_corners=False)
        target_aligned = self.kv_proj(target_aligned)      # (B, c_current, H, W)

        # --------------------------
        # Step 3: 准备Query, Key, Value
        # --------------------------
        # Query: 当前特征
        q = self.q_proj(current_feat)                      # (B, c_current, H, W)
        q = q.flatten(2).transpose(1, 2)                   # (B, N, c_current)
        q = self.norm_q(q)                                 # LayerNorm
        q = q.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)  # (B, H, D, N)

        # Key: 对齐后的目标特征
        k = target_aligned.flatten(2).transpose(1, 2)      # (B, N, c_current)
        k = self.norm_k(k)
        k = k.transpose(1, 2).view(B, self.num_heads, self.head_dim, N)  # (B, H, D, N)

        # Value: 对齐后的目标特征（不归一化）
        v = target_aligned.flatten(2).view(B, self.num_heads, self.head_dim, N)  # (B, H, D, N)

        # --------------------------
        # Step 4: 注意力计算
        # --------------------------
        # 对应位置点积（逐元素乘后求和），除以sqrt(head_dim)缩放
        logits = (q * k).sum(dim=2) / (self.head_dim ** 0.5)   # (B, H, N)

        # 类别频率正则：在logits上添加一个与类别分布相关的偏置（全局常数）
        if self.training and self.use_class_freq:
            reg_term = self.lambda_reg * self.log_inv_freq.mean().to(logits.device)
            logits = logits + reg_term

        # 注意力权重归一化
        attn_weights = logits.softmax(dim=-1)                  # (B, H, N)

        # 加权聚合Value
        fused = attn_weights.unsqueeze(2) * v                  # (B, H, D, N)

        # --------------------------
        # Step 5: 合并多头并恢复空间形状
        # --------------------------
        fused = fused.reshape(B, C, N)                         # (B, c_current, N)
        fused = fused.view(B, C, H, W)                         # (B, c_current, H, W)
        fused = self.out_proj(fused)                           # (B, c2, H, W)

        # 残差连接：要求c2 == c_current，否则无法相加
        # 此处假设c2 == c_current，设计时应确保
        return current_feat + fused
    

# class CS_AttnRes_Neck(nn.Module):
#     """
#     正确实现的长尾类别感知跨Stage注意力残差模块
#     核心逻辑：根据类别频率先验，动态调整目标特征的融合权重
#     - 融合C2细节时：给长尾类别更高权重，补全语义短板
#     - 融合C5语义时：给头部类别更高权重，降低误检
#     """
#     def __init__(self, c1, c2, is_detail_fusion=False, class_freq=None, lambda_reg=0.2):
#         """
#         Args:
#             c1: [目标特征通道数, 当前特征通道数]
#             c2: 输出通道数
#             is_detail_fusion: 是否是融合细节特征（C2），True=融合细节，False=融合语义
#             class_freq: 类别频率列表，按VisDrone官方顺序排列
#             lambda_reg: 长尾感知的强度系数，0=关闭，0.2=默认，0.5=强感知
#         """
#         super().__init__()
#         self.c_target = c1[0]   # 目标特征通道数（C2/C5）
#         self.c_current = c1[1]  # 当前特征通道数（P3/P4）
#         self.is_detail_fusion = is_detail_fusion
#         self.lambda_reg = lambda_reg

#         # --------------------------
#         # 长尾类别感知门控参数（核心创新）
#         # --------------------------
#         if class_freq is None:
#             class_freq = [1.0] * 10
#         # 预计算逆频率权重：长尾类别权重更高，头部类别权重更低
#         class_freq_tensor = torch.tensor(class_freq, dtype=torch.float32)
#         log_inv_freq = torch.log(1.0 / (class_freq_tensor + 1e-8))
#         self.register_buffer("class_weight", log_inv_freq)
#         # 可学习的门控系数：让模型自适应调整长尾感知的强度
#         self.gate = nn.Parameter(torch.tensor(1.0))

#         # --------------------------
#         # 投影层：保持之前的正确初始化
#         # --------------------------
#         self.q_proj = Conv(self.c_current, self.c_current, k=1, s=1, act=False)   # Query投影
#         self.kv_proj = Conv(self.c_target, self.c_current, k=1, s=1, act=False)  # Key/Value投影
#         self.out_proj = Conv(self.c_current, c2, k=1, s=1, act=False)              # 输出投影
        
#         # 只零初始化q_proj，保证初始状态等价于原生结构
#         nn.init.zeros_(self.q_proj.conv.weight)
#         if self.q_proj.conv.bias is not None:
#             nn.init.zeros_(self.q_proj.conv.bias)
        
#         # 归一化层
#         self.norm_q = nn.LayerNorm(self.c_current)
#         self.norm_k = nn.LayerNorm(self.c_current)

#     def forward(self, feat):
#         target_feat, current_feat = feat[0], feat[1]  # 解包：[目标特征(C2/C5), 当前特征(P3/P4)]
#         B, C, H, W = current_feat.shape
#         N = H * W

#         # --------------------------
#         # 1. 分辨率对齐 + 投影
#         # --------------------------
#         target_aligned = F.interpolate(target_feat, size=(H, W), mode='bilinear', align_corners=False)
#         target_aligned = self.kv_proj(target_aligned)

#         # --------------------------
#         # 2. 核心：长尾感知门控（真正的类别感知）
#         # --------------------------
#         # 计算类别权重的全局门控值
#         if self.is_detail_fusion:
#             # 融合细节特征（C2）：长尾类别权重更高，门控值更大
#             gate_weight = torch.sigmoid(self.gate * self.class_weight.mean()) * self.lambda_reg
#         else:
#             # 融合语义特征（C5）：头部类别权重更高，门控值更小（逆频率）
#             gate_weight = torch.sigmoid(self.gate * (1.0 / self.class_weight.mean())) * self.lambda_reg
        
#         # 直接作用于Key特征，调整目标特征的融合权重，路径极短
#         target_aligned = target_aligned * (1.0 + gate_weight) if self.is_detail_fusion else target_aligned * (1.0 - gate_weight)

#         # --------------------------
#         # 3. 轻量化单头注意力计算
#         # --------------------------
#         # 准备Q/K/V，简化冗余操作，保证训练速度
#         q = self.q_proj(current_feat).flatten(2).transpose(1, 2)  # (B, N, C)
#         q = self.norm_q(q).transpose(1, 2).view(B, C, N)          # (B, C, N)
        
#         k = target_aligned.flatten(2).transpose(1, 2)  # (B, N, C)
#         k = self.norm_k(k).transpose(1, 2).view(B, C, N)  # (B, C, N)
        
#         v = target_aligned.flatten(2).view(B, C, N)  # (B, C, N)

#         # 单头注意力计算，计算量最小
#         logits = torch.bmm(q.transpose(1, 2), k) / (C ** 0.5)
#         attn_weights = logits.softmax(dim=-1)
#         fused = torch.bmm(v, attn_weights.transpose(1, 2))

#         # --------------------------
#         # 4. 残差连接输出
#         # --------------------------
#         fused = fused.view(B, C, H, W)
#         fused = self.out_proj(fused)
#         return current_feat + fused


# class SmallTargetAttnRes(nn.Module):
#     """
#     小目标感知的深度注意力残差模块（BN版，支持前序通道数变化）
#     """
#     def __init__(self, c_current, num_blocks=4, block_idx=0, bn_momentum=0.1):
#         super().__init__()
#         self.c_current = c_current
#         self.num_blocks = num_blocks
#         self.block_idx = block_idx
        
#         # 伪查询向量，维度为当前通道数
#         self.pseudo_query = nn.Parameter(torch.zeros(c_current))
        
#         # 动态投影层字典：key 为输入通道数，value 为线性层
#         self.proj_layers = nn.ModuleDict()
        
#         # 全局平均池化（用于聚合前序特征的空间信息）
#         self.gap = nn.AdaptiveAvgPool2d(1)
        
#         # ECA核心：1D卷积实现通道交互（无降维）
#         self.conv1d = nn.Conv1d(1, 1, kernel_size=3, padding=3//2, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def _get_proj_layer(self, in_channels):
#         """获取或创建对应输入通道的线性投影层"""
#         key = str(in_channels)
#         if key not in self.proj_layers:
#             self.proj_layers[key] = nn.Linear(in_channels, self.c_current)
#         return self.proj_layers[key]

#     def forward(self, current_feat, prev_block_feats):
#         """
#         Args:
#             current_feat: [B, C_current, H, W]
#             prev_block_feats: 前序 Block 的压缩特征列表，每个元素 [B, C_i]
#         Returns:
#             out: [B, C_current, H, W]
#             compressed_feat: [B, C_current]
#         """
#         B, C, H, W = current_feat.shape
        
#         # 1. 压缩当前特征（用于后续传递）
#         compressed_feat = self.gap(current_feat).flatten(1)  # [B, C]
        
#         # 2. 第一个Block或无前序特征时直接返回
#         if self.block_idx == 0 or not prev_block_feats:
#             return current_feat, compressed_feat
        
#         # 3. 投影前序特征到统一维度（关键：保留原始投影值，不参与注意力计算）
#         proj_feats = []
#         for feat in prev_block_feats:
#             proj = self._get_proj_layer(feat.shape[1])(feat)
#             proj_feats.append(proj)
#         proj_stack = torch.stack(proj_feats, dim=0)  # [L, B, C]
        
#         # 4. 加权聚合：使用等权重平均（而非复杂的注意力权重）
#         #    因为前序特征已经被投影到同一空间，直接平均即可
#         attn_feat = proj_stack.mean(dim=0)  # [B, C]
        
#         # 5. ECA通道注意力生成
#         #    将[B, C] -> [B, 1, C] 适配Conv1d
#         attn_feat_expanded = attn_feat.unsqueeze(1)  # [B, 1, C]
#         channel_weights = self.conv1d(attn_feat_expanded)  # [B, 1, C]
#         channel_weights = self.sigmoid(channel_weights.squeeze(1))  # [B, C]
        
#         # 6. 应用通道注意力（残差形式）
#         channel_weights = channel_weights.view(B, C, 1, 1)
#         out = current_feat + current_feat * channel_weights
#         # out = current_feat * channel_weights
        
#         return out, compressed_feat
    

class ProgressiveSpatialAttnRes(nn.Module):
    """
    渐进式空间注意力残差模块
    通过逐阶段融合前序特征，避免一次性聚合导致的后期退化
    """
    def __init__(self, c_current, num_blocks=4, block_idx=0, bn_momentum=0.1):
        super().__init__()
        self.c_current = c_current
        self.num_blocks = num_blocks
        self.block_idx = block_idx

        # 动态投影层字典（同原版）
        self.proj_layers = nn.ModuleDict()
        self.bn = nn.BatchNorm1d(c_current, momentum=bn_momentum)

        # 空间注意力生成器（可共享权重，也可独立）
        # 这里使用共享权重，减少参数量且更易训练
        self.spatial_conv = nn.Conv2d(c_current, 1, kernel_size=1)
        nn.init.zeros_(self.spatial_conv.weight)
        nn.init.zeros_(self.spatial_conv.bias)

        # 可选的层归一化，用于稳定逐阶段融合
        self.layer_norm = nn.LayerNorm(c_current)

        # 可学习的温度系数，控制注意力锐度（防止后期极端化）
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def _get_proj_layer(self, in_channels):
        key = str(in_channels)
        if key not in self.proj_layers:
            self.proj_layers[key] = nn.Linear(in_channels, self.c_current)
        return self.proj_layers[key]

    def forward(self, current_feat, prev_block_feats):
        """
        Args:
            current_feat: [B, C, H, W]
            prev_block_feats: list of [B, C_i] (C_i可不同)
        Returns:
            out: [B, C, H, W]
            compressed_feat: [B, C]
        """
        B, C, H, W = current_feat.shape
        compressed_feat = F.adaptive_avg_pool2d(current_feat, (1, 1)).flatten(1)

        # 若无前序特征，直接返回
        if self.block_idx == 0 or not prev_block_feats:
            return current_feat, compressed_feat

        # 初始特征为当前特征
        out = current_feat

        # 按顺序逐个融合前序特征
        for feat in prev_block_feats:
            # 投影前序特征
            proj = self._get_proj_layer(feat.shape[1])(feat)  # [B, C]
            # 广播到空间尺寸
            attn_broadcast = proj.view(B, C, 1, 1).expand(B, C, H, W)

            # 生成空间注意力图
            spatial_mask = torch.sigmoid(self.spatial_conv(attn_broadcast) / self.temperature)  # [B,1,H,W]

            # 调制当前特征
            delta = out * spatial_mask

            # 残差融合
            out = out + delta

            # 可选：层归一化稳定特征分布
            # out = out.permute(0,2,3,1).contiguous()
            # out = self.layer_norm(out).permute(0,3,1,2)

        # 最终输出
        return out, compressed_feat
    

class AttnResModule2D(nn.Module):
    """
    修正版：避免批归一化导致注意力权重退化
    """
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # 查询投影
        self.q_proj = Conv(dim, dim, k=1, s=1, act=False)
        
        # 键值归一化：使用 LayerNorm 在通道维度归一化，且对每个前序块独立
        self.norm_kv = nn.LayerNorm(dim)
        
        # 可学习温度系数（初始为1，但可训练）
        self.temperature = nn.Parameter(torch.ones(1))
        
        # 输出投影（可选）
        # self.out_proj = Conv(dim, dim, k=1, s=1, act=False) if True else nn.Identity()

    def forward(self, prev_blocks, current_x):
        if not prev_blocks:
            return current_x
        
        B, C, H, W = current_x.shape
        N = len(prev_blocks)
        
        # 堆叠前序块 (N, B, C, H, W)
        V = torch.stack(prev_blocks, dim=0)   # (N, B, C, H, W)
        
        # 关键修改：对每个前序块独立归一化通道维度
        # 将 (N, B, C, H, W) 转为 (N*B, C, H, W) 无法独立归一化，因此改用 Permute
        # 方法：将 (N, B, C, H, W) -> (N*B, C, H, W) 然后应用 InstanceNorm2d，但 InstanceNorm 会独立处理每个样本
        # 更简单：将通道维度移到末尾，使用 LayerNorm
        V_perm = V.permute(0, 1, 3, 4, 2)    # (N, B, H, W, C)
        V_norm = self.norm_kv(V_perm)          # 对最后一维 (C) 归一化
        V_norm = V_norm.permute(0, 1, 4, 2, 3) # (N, B, C, H, W)
        
        # 全局平均池化得到 (N, B, C)
        K_gap = V_norm.mean(dim=(3, 4))        # (N, B, C)
        V_gap = V.mean(dim=(3, 4))             # (N, B, C)   # 使用原始值聚合
        
        # 查询：当前特征全局池化后投影
        q = self.q_proj(current_x)             # (B, C, H, W)
        q_gap = q.mean(dim=(2, 3))             # (B, C)
        
        # 多头拆分（若 num_heads > 1）
        if self.num_heads > 1:
            q_gap = q_gap.view(B, self.num_heads, self.head_dim)
            K_gap = K_gap.view(N, B, self.num_heads, self.head_dim)
            # 计算注意力分数 (B, N, num_heads)
            logits = torch.einsum('b h d, n b h d -> b n h', q_gap, K_gap) / (self.head_dim ** 0.5)
            attn_weights = logits.softmax(dim=1)   # (B, N, num_heads)
            # 加权聚合
            V_gap = V_gap.view(N, B, self.num_heads, self.head_dim)
            aggregated = torch.einsum('b n h, n b h d -> b h d', attn_weights, V_gap)
            aggregated = aggregated.reshape(B, C)
        else:
            # 单头，使用温度系数调节 softmax 锐度
            logits = torch.einsum('bc, nbc -> bn', q_gap, K_gap) / (self.head_dim ** 0.5)
            logits = logits / self.temperature   # 可学习温度
            attn_weights = logits.softmax(dim=1)  # (B, N)
            aggregated = torch.einsum('bn, nbc -> bc', attn_weights, V_gap)  # (B, C)
        
        # 广播到空间，残差相加
        aggregated = aggregated.view(B, C, 1, 1)
        out = q + aggregated   # 直接相加（也可用可学习缩放）
        
        # 输出投影
        # out = self.out_proj(out)
        
        return out


class SpatialAttnResModule2D(nn.Module):
    """
    空间感知的2D注意力残差模块（改进版）
    使用自适应池化保留空间结构，并在每个空间位置独立计算跨块注意力。
    适用于小目标检测，避免全局池化丢失细节。
    """
    def __init__(self, dim, reduction=4, spatial_size=4, use_spatial_query=True):
        """
        Args:
            dim: 输入通道数
            reduction: MLP 降维比率，None 表示不使用 MLP
            spatial_size: 自适应池化后的空间尺寸 (P x P)
            use_spatial_query: 是否使用空间维度的 Query（建议 True）
        """
        super().__init__()
        self.dim = dim
        self.spatial_size = spatial_size
        self.use_spatial_query = use_spatial_query
        
        # 查询投影：1x1 卷积，将当前输入映射为 Query（空间维度保留）
        self.q_proj = Conv(dim, dim, k=1, s=1, act=False)
        
        # 键值归一化：LayerNorm 在通道维度，但键值形状为 (N, B, C, P, P)
        # 我们将最后一维（空间）展开到 Batch 维度进行归一化，或使用 GroupNorm
        # 这里使用 LayerNorm 在通道维度，对每个空间位置独立归一化
        self.norm_kv = nn.LayerNorm(dim)
        
        # 可选的非线性变换
        if reduction is not None:
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(dim // reduction, dim, bias=False)
            )
        else:
            self.mlp = nn.Identity()
        
        # 可学习的空间位置编码，形状 (P, P, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, dim, spatial_size, spatial_size) * 0.02)
        
        # 可学习残差缩放因子
        self.alpha = nn.Parameter(torch.tensor(0.0))
        
        # 输出投影
        self.out_proj = Conv(dim, dim, k=1, s=1, act=False)
        
    def forward(self, prev_blocks, current_x):
        """
        Args:
            prev_blocks: List of previous Block representations, each (B, C, H, W)
            current_x: Current Block input feature (B, C, H, W)
        Returns:
            aggregated_x: 聚合后的输入特征 (B, C, H, W)
        """
        # if not prev_blocks:
        #     return current_x
        
        B, C, H, W = current_x.shape
        N = len(prev_blocks)
        P = self.spatial_size
        
        # ========== 1. 构建键值（保留空间结构） ==========
        # 将所有前序块堆叠，并自适应池化到固定空间尺寸
        prev_stack = torch.stack(prev_blocks, dim=0)  # (N, B, C, H, W)
        # 将 (N, B, C, H, W) 合并为 (N*B, C, H, W)
        prev_flat = prev_stack.view(N * B, C, H, W)
        # 自适应池化
        kv_pooled_flat = F.adaptive_avg_pool2d(prev_flat, (P, P))  # (N*B, C, P, P)
        # 恢复形状为 (N, B, C, P, P)
        kv_pooled = kv_pooled_flat.view(N, B, C, P, P)
        
        # 添加空间位置编码
        kv_pooled = kv_pooled + self.pos_embed.unsqueeze(0)  # (N, B, C, P, P)
        
        # 转置形状以便在通道维度归一化：(N, B, P, P, C)
        kv_pooled = kv_pooled.permute(0, 1, 3, 4, 2)  # (N, B, P, P, C)
        kv_pooled = self.norm_kv(kv_pooled)            # (N, B, P, P, C)
        keys = kv_pooled                               # (N, B, P, P, C)
        values = kv_pooled
        
        # ========== 2. 构建 Query ==========
        if self.use_spatial_query:
            # 对当前输入同样进行自适应池化，得到 (B, C, P, P)
            q_pooled = F.adaptive_avg_pool2d(current_x, (P, P))  # (B, C, P, P)
            # 通过 Query 投影（1x1 卷积）
            q = self.q_proj(q_pooled)                  # (B, C, P, P)
            # 添加位置编码
            q = q + self.pos_embed
            # 转置为 (B, P, P, C) 便于计算
            q = q.permute(0, 2, 3, 1)                  # (B, P, P, C)
        else:
            # 兼容原版：全局池化后投影
            q_pooled = current_x.mean(dim=(2, 3))      # (B, C)
            q = self.q_proj(q_pooled.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)  # (B, C)
            q = q.unsqueeze(-1).unsqueeze(-1)          # (B, C, 1, 1)
            q = q.permute(0, 2, 3, 1)                  # (B, 1, 1, C)
            # 扩展至 (B, P, P, C) 以便统一计算
            q = q.expand(-1, P, P, -1)
        
        # ========== 3. 注意力计算：每个空间位置独立跨块聚合 ==========
        # 我们希望：对于每个空间位置 (i,j)，计算该位置 Query 与所有历史块同一位置 Key 的注意力
        # 即: logits_{b,n,i,j} = sum_c (q_{b,i,j,c} * keys_{n,b,i,j,c}) / sqrt(C)
        # 通过 einsum 实现: 'bnijc, bijnc -> bnij' 注意维度顺序调整
        # 当前形状: q: (B, P, P, C), keys: (N, B, P, P, C)
        # 将 q 扩展维度: (B, P, P, 1, C) 与 keys (N, B, P, P, C) 在最后一个维度点积
        # 更简洁: 先将 q 转置为 (B, P, P, C)，keys 保持 (N, B, P, P, C)
        # 计算点积: (B, P, P, C) * (N, B, P, P, C) -> 对 C 求和得到 (B, P, P, N)
        # 但需要维度对齐，使用 einsum: 'b i j c, n b i j c -> b i j n'
        logits = torch.einsum('b i j c, n b i j c -> b i j n', q, keys) / math.sqrt(C)
        # logits shape: (B, P, P, N)
        attn_weights = F.softmax(logits, dim=-1)      # 对最后一个维度（块维度）softmax
        
        # 加权聚合 values: (B, P, P, N) * (N, B, P, P, C) -> (B, P, P, C)
        aggregated = torch.einsum('b i j n, n b i j c -> b i j c', attn_weights, values)
        
        # ========== 4. 非线性变换 ==========
        aggregated = self.mlp(aggregated)             # (B, P, P, C)
        
        # ========== 5. 上采样回原空间尺寸 ==========
        aggregated = aggregated.permute(0, 3, 1, 2)   # (B, C, P, P)
        aggregated = F.interpolate(aggregated, size=(H, W), mode='bilinear', align_corners=False)
        
        # ========== 6. 注入当前特征 ==========
        out = current_x + self.alpha * aggregated
        
        # ========== 7. 输出投影 ==========
        out = self.out_proj(out)
        
        return out


class DenseMultiScaleAlign(nn.Module):
    """
    DenseNet风格多尺度密集特征对齐（去除Focus分支版）：
    1. 彻底去除Focus分支
    2. 保留多尺度密集采样：平均池化、最大池化、双线性插值 3种方式
    3. 直接调用你提供的ChannelAttention做自适应融合
    4. 100%兼容Ultralytics原生组件
    """
    def __init__(self, in_channels, out_channels, target_stride_ratio=2):
        super().__init__()
        self.target_ratio = target_stride_ratio
        self.out_channels = out_channels

        # --------------------------
        # 1. 3种下采样方式的投影层（全用Ultralytics原生Conv）
        # --------------------------
        # 平均池化：保留全局上下文
        self.avg_pool = nn.AvgPool2d(kernel_size=target_stride_ratio, stride=target_stride_ratio)
        self.avg_pool_proj = Conv(in_channels, out_channels, k=1, act=True)
        # 最大池化：保留边缘和角点（小目标关键）
        self.max_pool = nn.MaxPool2d(kernel_size=target_stride_ratio, stride=target_stride_ratio)
        self.max_pool_proj = Conv(in_channels, out_channels, k=1, act=True)
        # 双线性插值：保留空间连续性
        self.bilinear_proj = Conv(in_channels, out_channels, k=1, act=True)

        # --------------------------
        # 2. 【直接调用】你提供的ChannelAttention做自适应融合
        # --------------------------
        self.fusion_attn = ChannelAttention(channels=out_channels * 3)

        self.final_proj = Conv(out_channels * 3, out_channels, k=1, act=True)

    def forward(self, x, target_size):
        """
        x: 输入的原始浅层特征 [B, C, H, W]
        target_size: 目标尺寸 (H, W)
        """
        B, C, H, W = x.shape
        target_H, target_W = target_size

        # --------------------------
        # 1. 3种方式密集下采样
        # --------------------------
        # 平均池化
        x_avg = F.adaptive_avg_pool2d(x, target_size)
        x_avg = self.avg_pool_proj(x_avg)
        
        # 最大池化
        x_max = F.adaptive_max_pool2d(x, target_size)
        x_max = self.max_pool_proj(x_max)
        
        # 双线性插值
        x_bilinear = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
        x_bilinear = self.bilinear_proj(x_bilinear)

        # --------------------------
        # 2. 拼接3种方式的特征
        # --------------------------
        concat_feat = torch.cat([x_avg, x_max, x_bilinear], dim=1)

        # --------------------------
        # 3. 【直接调用】你提供的ChannelAttention做自适应融合
        # --------------------------
        # fused_feat = self.fusion_attn(concat_feat)

        # --------------------------
        # 4. 【核心修改】1×1卷积降维（替代切片相加）
        # --------------------------
        final_feat = self.final_proj(concat_feat)
        
        return final_feat


class SpatialAttnRes(nn.Module):
    """
    空间注意力残差模块：将跨层特征展平为序列，在空间维度进行注意力交互，
    实现跨层、跨空间的自适应特征融合。参考 VVP 的空间相似度计算思想。
    """
    def __init__(self, channels, use_cosine_sim=True):
        super().__init__()
        self.channels = channels
        self.use_cosine_sim = use_cosine_sim

        # QKV 投影（保持原设计的可学习投影）
        self.q_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self.eps = 1e-8
        # 缩放因子（余弦相似度模式无需额外缩放）
        self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

    def forward(self, x, history_feats):
        """
        Args:
            x: 当前特征 [B, C, H, W]
            history_feats: 对齐后的历史特征列表，每个 [B, C, H, W]
        Returns:
            融合特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        N = H * W                     # 空间位置总数

        # 1. 投影当前特征为 Query，并展平为 [B, N, C]
        q = self.q_proj(x).view(B, C, N).transpose(1, 2)   # [B, N, C]

        # 2. 所有特征（历史 + 当前）投影为 Key 和 Value，并在空间维度拼接
        all_feats = history_feats + [x]
        k_list, v_list = [], []
        for feat in all_feats:
            k = self.k_proj(feat).view(B, C, N)            # [B, C, N]
            v = self.v_proj(feat).view(B, C, N)
            k_list.append(k)
            v_list.append(v)

        # 在空间维度拼接： [B, C, K*N] -> transpose -> [B, K*N, C]
        k_all = torch.cat(k_list, dim=2).transpose(1, 2)   # [B, K*N, C]
        v_all = torch.cat(v_list, dim=2).transpose(1, 2)   # [B, K*N, C]

        # 3. 计算空间相似度（借鉴 VVP 的特征归一化 + 点积）
        if self.use_cosine_sim:
            q = F.normalize(q, p=2, dim=-1)                # [B, N, C]
            k_all = F.normalize(k_all, p=2, dim=-1)        # [B, K*N, C]
            attn = torch.bmm(q, k_all.transpose(1, 2))    # [B, N, K*N]
        else:
            attn = torch.bmm(q, k_all.transpose(1, 2)) / self.scale  # [B, N, K*N]

        # 4. 在空间维度（dim=-1）上做 softmax 获得空间注意力权重
        attn_weights = F.softmax(attn, dim=-1)             # [B, N, K*N]

        # 5. 加权聚合 Value
        attn_out = torch.bmm(attn_weights, v_all)          # [B, N, C]
        attn_out = attn_out.transpose(1, 2).view(B, C, H, W)  # [B, C, H, W]

        # 6. 残差连接 + 输出投影
        out = x + self.out_proj(attn_out)
        return out
    

def dwt_init(x):
    """
    使用 Haar 小波进行下采样 (Stride=2)
    将 [B, C, H, W] 转换为 [B, 4C, H/2, W/2]
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    # 分别对应 LL, LH, HL, HH
    return torch.cat([x1 + x2 + x3 + x4, x1 - x2 + x3 - x4, x1 + x2 - x3 - x4, x1 - x2 - x3 + x4], dim=1)


class WaveletDownsample(nn.Module):
    """
    小波下采样层：减少尺寸的同时，通过 1x1 卷积压缩频率维度带来的通道增长
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2 = Conv(in_channels, out_channels, k=3, s=2)
        self.conv = Conv(in_channels * 4, out_channels, k=1)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, 4C, H/2, W/2]
        identity = x
        x = dwt_init(x)
        return self.conv(x) + self.conv2(identity)


class WaveletDownsample1(nn.Module):
    """
    小波下采样层：减少尺寸的同时，通过 1x1 卷积压缩频率维度带来的通道增长
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv(in_channels * 4, out_channels, k=1)

    def forward(self, x):
        # x: [B, C, H, W] -> [B, 4C, H/2, W/2]
        x = dwt_init(x)
        return self.conv(x)


class WaveletDownsample2(nn.Module):
    """
    Wavelet Rectifying Downsampling (WRD) 模块.
    利用 Haar 小波提取的高频分量 (LH, HL, HH) 修正最大池化下采样的混叠失真。
    
    输入: x, shape (B, C, H, W)   (H, W 需为偶数)
    输出: out, shape (B, C, H/2, W/2)
    """
    def __init__(self, in_channels, out_channels):
        super(WaveletDownsample2, self).__init__()
        # 从 dwt_init 中提取 LH, HL, HH，共 3*in_channels
        # 第一个卷积: 3*C -> C, 3x3, BN+ReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(3 * in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        # 第二个卷积: C -> C, 3x3, BN (无激活)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels)
        )
        self.conv3 = Conv(in_channels, out_channels, k=1)  # 用于最大池化分支的通道调整
        # 最大池化下采样 (stride=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # 1. Haar 小波分解
        wave_all = dwt_init(x)            # (B, 4C, H/2, W/2)
        # 分离四个子带: LL, LH, HL, HH 各 (B, C, H/2, W/2)
        C = x.shape[1]
        LL = wave_all[:, 0:C, :, :]       # 低频
        LH = wave_all[:, C:2*C, :, :]     # 水平高频
        HL = wave_all[:, 2*C:3*C, :, :]   # 垂直高频
        HH = wave_all[:, 3*C:4*C, :, :]   # 对角高频
        # 2. 拼接高频子带 -> (B, 3C, H/2, W/2)
        high_freq = torch.cat([LH, HL, HH], dim=1)
        # 3. 卷积修正分支
        Ft = self.conv1(high_freq)   # (B, C, H/2, W/2)
        Ft = self.conv2(Ft)          # 保持通道
        # 4. 最大池化下采样
        pooled = self.pool(x)        # (B, C, H/2, W/2)
        # 5. 修正：相加
        out = pooled + Ft
        out = self.conv3(out)        # 调整通道到 out_channels
        return out

class HierarchicalWaveletAlign(nn.Module):
    """
    层级小波对齐模块：用于 Stride=8 的场景 (如 P2 -> P5)
    通过 3 次小波变换平滑降采样
    """
    def __init__(self, c1, c2, steps=3):
        super().__init__()
        self.steps = steps
        modules = []
        curr_c = c1
        for i in range(steps):
            # 每一步下采样 2 倍，并逐渐调整通道数
            next_c = c2 if i == steps - 1 else max(c1, c2) 
            modules.append(WaveletDownsample(curr_c, next_c))
            curr_c = next_c
        self.stages = nn.Sequential(*modules)

    def forward(self, x):
        return self.stages(x)

# 原版
class AttnRes(nn.Module):
    """
    通道维度注意力残差模块：对跨层特征在通道维度做 softmax 激活，
    实现自适应特征融合。无额外存储，轻量化。
    """
    def __init__(self, channels, use_cosine_sim=True):
        super().__init__()
        self.channels = channels
        self.use_cosine_sim = use_cosine_sim

        self.q_proj = Conv(channels, channels, k=1)
        self.k_proj = Conv(channels, channels, k=1)
        self.v_proj = Conv(channels, channels, k=1)
        self.out_proj = Conv(channels, channels, k=1)

        # 数值稳定
        self.eps = 1e-8

        # 点积分支的缩放因子
        self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

    def forward(self, x, history_feats):
        """
        x: 当前特征 [B, C, H, W]
        history_feats: 对齐后的历史特征列表，每个 [B, C, H, W]
        输出: 融合后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        K = len(history_feats) + 1          # 总层数（历史+当前）

        # 分别投影每个特征，然后拼接
        # Query: 当前特征
        q = self.q_proj(x)                  # [B, C, H, W]

        # Key: 所有特征（历史+当前）分别投影后拼接
        k_list = []
        v_list = []
        for feat in history_feats + [x]:
            k_list.append(self.k_proj(feat))   # [B, C, H, W]
            v_list.append(self.v_proj(feat))   # [B, C, H, W]
        k = torch.stack(k_list, dim=1)         # [B, K, C, H, W]
        v = torch.stack(v_list, dim=1)         # [B, K, C, H, W]

        # 相似度计算（Query 与 Key 在所有层间）
        if self.use_cosine_sim:
            q_norm = q / (q.norm(p=2, dim=1, keepdim=True) + self.eps)   # [B, C, H, W]
            k_norm = k / (k.norm(p=2, dim=2, keepdim=True) + self.eps)   # [B, K, C, H, W]
            # 点积: [B, C, H, W]
            attn_logits = (q_norm.unsqueeze(1) * k_norm).sum(dim=1)
        else:
            attn_logits = torch.einsum('b c h w, b k c h w -> b c h w', q, k) / self.scale

        # 关键：在通道维度（C）上做 softmax，每个空间位置独立
        attn_weights = F.softmax(attn_logits, dim=1)       # [B, C, H, W]

        # 加权聚合 Value
        attn_weights_expand = attn_weights.unsqueeze(1)    # [B, 1, C, H, W]
        attn_out = (attn_weights_expand * v).sum(dim=1)    # [B, C, H, W]

        # 输出投影
        out = x + self.out_proj(attn_out)
        return out

# 0823版本
# class AttnRes(nn.Module):
#     """
#     通道维度注意力残差模块：对跨层特征在通道维度做 softmax 激活，
#     实现自适应特征融合。无额外存储，轻量化。
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim

#         self.q_proj = Conv(channels, channels, k=1)
#         self.k_proj = Conv(channels, channels, k=1)
#         self.v_proj = Conv(channels, channels, k=1)
#         self.out_proj = Conv(channels, channels, k=1)

#         # 数值稳定
#         self.eps = 1e-8
#         # 初始 3.0 让权重一开始就有区分度，训练中可学习
#         self.tau = nn.Parameter(torch.tensor(1.0))

#         # 点积分支的缩放因子
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#     def forward(self, x, history_feats):
#         """
#         x: 当前特征 [B, C, H, W]
#         history_feats: 对齐后的历史特征列表，每个 [B, C, H, W]
#         输出: 融合后的特征 [B, C, H, W]
#         """
#         B, C, H, W = x.shape
#         K = len(history_feats) + 1          # 总层数（历史+当前）

#         # 分别投影每个特征，然后拼接
#         # Query: 当前特征
#         q = self.q_proj(x)                  # [B, C, H, W]

#         # Key: 所有特征（历史+当前）分别投影后拼接
#         k_list = []
#         v_list = []
#         for feat in history_feats + [x]:
#             k_list.append(self.k_proj(feat))   # [B, C, H, W]
#             v_list.append(self.v_proj(feat))   # [B, C, H, W]
#         k = torch.stack(k_list, dim=1)         # [B, K, C, H, W]
#         v = torch.stack(v_list, dim=1)         # [B, K, C, H, W]

#         # 相似度计算（Query 与 Key 在所有层间）
#         if self.use_cosine_sim:
#             q_norm = q / (q.norm(p=2, dim=1, keepdim=True) + self.eps)   # [B, C, H, W]
#             k_norm = k / (k.norm(p=2, dim=2, keepdim=True) + self.eps)   # [B, K, C, H, W]
#             # 点积: [B, C, H, W]
#             attn_logits = (q_norm.unsqueeze(1) * k_norm).sum(dim=1)
#         else:
#             attn_logits = torch.einsum('b c h w, b k c h w -> b c h w', q, k) / self.scale

#         # 关键：在通道维度（C）上做 softmax，每个空间位置独立
#         attn_weights = F.softmax(attn_logits * self.tau.clamp(min=0.1, max=8.0), dim=1)       # [B, C, H, W]

#         # 加权聚合 Value
#         attn_weights_expand = attn_weights.unsqueeze(1)    # [B, 1, C, H, W]
#         attn_out = (attn_weights_expand * v).sum(dim=1)    # [B, C, H, W]

#         # 输出投影
#         out = x + self.out_proj(attn_out)
#         return out
    
# 0820    n尺度没有任何提升
# class AttnRes(nn.Module):
#     """
#     通道维度注意力残差模块：对跨层特征在通道维度做 softmax 激活，
#     实现自适应特征融合。无额外存储，轻量化。
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim

#         self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

#         # 数值稳定
#         self.eps = 1e-8

#         # 点积分支的缩放因子
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#     def forward(self, x, history_feats):
#         """
#         x: 当前特征 [B, C, H, W]
#         history_feats: 对齐后的历史特征列表，每个 [B, C, H, W]
#         输出: 融合后的特征 [B, C, H, W]
#         """
#         B, C, H, W = x.shape
#         K = len(history_feats) + 1          # 总层数（历史+当前）

#         # 分别投影每个特征，然后拼接
#         # Query: 当前特征
#         q = self.q_proj(x)                  # [B, C, H, W]

#         # Key: 所有特征（历史+当前）分别投影后拼接
#         k_list = []
#         v_list = []
#         for feat in history_feats + [x]:
#             k_list.append(self.k_proj(feat))   # [B, C, H, W]
#             v_list.append(self.v_proj(feat))   # [B, C, H, W]
#         k = torch.stack(k_list, dim=1)         # [B, K, C, H, W]
#         v = torch.stack(v_list, dim=1)         # [B, K, C, H, W]

#         # 相似度计算（Query 与 Key 在所有层间）
#         if self.use_cosine_sim:
#             q_norm = q / (q.norm(p=2, dim=1, keepdim=True) + self.eps)   # [B, C, H, W]
#             k_norm = k / (k.norm(p=2, dim=2, keepdim=True) + self.eps)   # [B, K, C, H, W]
#             # 点积: [B, C, H, W]
#             attn_logits = (q_norm.unsqueeze(1) * k_norm).sum(dim=1)
#         else:
#             attn_logits = torch.einsum('b c h w, b k c h w -> b c h w', q, k) / self.scale

#         # 关键：在通道维度（C）上做 softmax，每个空间位置独立
#         attn_weights = F.softmax(attn_logits, dim=1)       # [B, C, H, W]

#         # 加权聚合 Value
#         attn_weights_expand = attn_weights.unsqueeze(1)    # [B, 1, C, H, W]
#         attn_out = (attn_weights_expand * v).sum(dim=1)    # [B, C, H, W]

#         # 输出投影
#         out = x + self.out_proj(attn_out)
#         return out
    
# 0817版本
# class AttnRes(nn.Module):
#     """
#     Cross-layer attention: per-pixel softmax over K layers.
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#         # 初始 3.0 让权重一开始就有区分度，训练中可学习
#         self.tau = nn.Parameter(torch.tensor(3.0))

#         self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.out_proj = nn.Conv2d(channels, channels, 1)

#         # 小方差非零初始化：残差接近恒等，但 q/k/v/tau 从一开始就有梯度
#         nn.init.normal_(self.out_proj.weight, std=0.01)
#         nn.init.zeros_(self.out_proj.bias)

#     def forward(self, x, history_feats):
#         all_feats = history_feats

#         q = self.q_proj(x)                                          # [B, C, H, W]
#         k = torch.stack([self.k_proj(f) for f in all_feats], dim=1) # [B, K, C, H, W]
#         v = torch.stack([self.v_proj(f) for f in all_feats], dim=1) # [B, K, C, H, W]

#         if self.use_cosine_sim:
#             q = F.normalize(q, p=2, dim=1)
#             k = F.normalize(k, p=2, dim=2)
#             # 对全部通道求和，得到每个空间位置、每个层的相似度
#             logits = (q.unsqueeze(1) * k).sum(dim=2)               # [B, K, H, W]
#         else:
#             logits = torch.einsum('b c h w, b k c h w -> b k h w', q, k) / self.scale

#         # 在 K（层）维度上 softmax，每个空间位置得到一组跨层权重
#         attn_weights = F.softmax(logits * self.tau.clamp(min=0.1, max=8.0), dim=1)
#         attn_out = (attn_weights.unsqueeze(2) * v).sum(dim=1)      # [B, C, H, W]

#         return x + self.out_proj(attn_out)
    
# 0822版本 ,没有self.tau参数效果非常不好
# class AttnRes(nn.Module):
#     """
#     Cross-layer attention: per-pixel softmax over K layers.
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#         # 初始 3.0 让权重一开始就有区分度，训练中可学习
#         # self.tau = nn.Parameter(torch.tensor(3.0))

#         self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.out_proj = nn.Conv2d(channels, channels, 1)

#         # 小方差非零初始化：残差接近恒等，但 q/k/v/tau 从一开始就有梯度
#         nn.init.normal_(self.out_proj.weight, std=0.01)
#         nn.init.zeros_(self.out_proj.bias)

#     def forward(self, x, history_feats):
#         all_feats = history_feats

#         q = self.q_proj(x)                                          # [B, C, H, W]
#         k = torch.stack([self.k_proj(f) for f in all_feats], dim=1) # [B, K, C, H, W]
#         v = torch.stack([self.v_proj(f) for f in all_feats], dim=1) # [B, K, C, H, W]

#         if self.use_cosine_sim:
#             q = F.normalize(q, p=2, dim=1)
#             k = F.normalize(k, p=2, dim=2)
#             # 对全部通道求和，得到每个空间位置、每个层的相似度
#             logits = (q.unsqueeze(1) * k).sum(dim=2)               # [B, K, H, W]
#         else:
#             logits = torch.einsum('b c h w, b k c h w -> b k h w', q, k) / self.scale

#         # 在 K（层）维度上 softmax，每个空间位置得到一组跨层权重
#         attn_weights = F.softmax(logits, dim=1)
#         attn_out = (attn_weights.unsqueeze(2) * v).sum(dim=1)      # [B, C, H, W]

#         return x + self.out_proj(attn_out)

# 0824版本，01版历史特征没有x；02版历史特征有x，效果非常不好；03版本使用Conv取代Conv2d，效果一般，但是m尺度还可以；
# class AttnRes(nn.Module):
#     """
#     Cross-layer attention: per-pixel softmax over K layers.
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#         # 初始 3.0 让权重一开始就有区分度，训练中可学习
#         # self.tau = nn.Parameter(torch.tensor(3.0))

#         self.q_proj = Conv(channels, channels, 1)
#         self.k_proj = Conv(channels, channels, 1)
#         self.v_proj = Conv(channels, channels, 1)
#         self.out_proj = Conv(channels, channels, 1)

#         # 小方差非零初始化：残差接近恒等，但 q/k/v/tau 从一开始就有梯度
#         # nn.init.normal_(self.out_proj.weight, std=0.01)
#         # nn.init.zeros_(self.out_proj.bias)

#     def forward(self, x, history_feats):
#         all_feats = history_feats
        
#         q = self.q_proj(x)                          # [B, C, H, W]
#         k = sum(self.k_proj(f) for f in all_feats)  # [B, C, H, W]
#         v = sum(self.v_proj(f) for f in all_feats)  # [B, C, H, W]

#         if self.use_cosine_sim:
#             q = F.normalize(q, p=2, dim=1)
#             k = F.normalize(k, p=2, dim=1)
#             # 对全部通道求和，得到每个空间位置、每个层的相似度
#             logits = q * k               # [B, C, H, W]
#         else:
#             logits = torch.einsum('b c h w, b k c h w -> b k h w', q, k) / self.scale

#         # 在 C（通道）维度上 softmax，每个空间位置得到一组跨层权重
#         attn_weights = F.softmax(logits, dim=1)
#         attn_out = attn_weights * v      # [B, C, H, W]

#         return x + self.out_proj(attn_out)


# 0825版本，效果一般
# class AttnRes(nn.Module):
#     """
#     Cross-layer attention: per-pixel softmax over K layers.
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#         # 初始 3.0 让权重一开始就有区分度，训练中可学习
#         # self.tau = nn.Parameter(torch.tensor(3.0))

#         self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
#         self.out_proj = nn.Conv2d(channels, channels, 1)

#         # 小方差非零初始化：残差接近恒等，但 q/k/v/tau 从一开始就有梯度
#         nn.init.normal_(self.out_proj.weight, std=0.01)
#         nn.init.zeros_(self.out_proj.bias)

#     def forward(self, x, history_feats):
#         all_feats = history_feats
        
#         q = self.q_proj(x)                          # [B, C, H, W]
#         k = self.k_proj(sum(all_feats))  # [B, C, H, W]
#         v = self.v_proj(sum(all_feats))  # [B, C, H, W]

#         if self.use_cosine_sim:
#             q = F.normalize(q, p=2, dim=1)
#             k = F.normalize(k, p=2, dim=1)
#             # 对全部通道求和，得到每个空间位置、每个层的相似度
#             logits = q * k               # [B, C, H, W]
#         else:
#             logits = torch.einsum('b c h w, b k c h w -> b k h w', q, k) / self.scale

#         # 在 C（通道）维度上 softmax，每个空间位置得到一组跨层权重
#         attn_weights = F.softmax(logits, dim=1)
#         attn_out = attn_weights * v      # [B, C, H, W]

#         return x + self.out_proj(attn_out)

# 0819版本
# class AttnRes(nn.Module):
#     """
#     Cross-layer attention: per-pixel softmax over K layers.
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#         # 初始 3.0 让权重一开始就有区分度，训练中可学习
#         self.tau = nn.Parameter(torch.tensor(3.0))

#         self.q_proj = Conv(channels, channels, 1)
#         self.k_proj = Conv(channels, channels, 1)
#         self.v_proj = Conv(channels, channels, 1)
#         self.out_proj = Conv(channels, channels, 1)

#     def forward(self, x, history_feats):
#         all_feats = history_feats 

#         q = self.q_proj(x)                                          # [B, C, H, W]
#         k = torch.stack([self.k_proj(f) for f in all_feats], dim=1) # [B, K, C, H, W]
#         v = torch.stack([self.v_proj(f) for f in all_feats], dim=1) # [B, K, C, H, W]

#         if self.use_cosine_sim:
#             q = F.normalize(q, p=2, dim=1)
#             k = F.normalize(k, p=2, dim=2)
#             # 对全部通道求和，得到每个空间位置、每个层的相似度
#             logits = (q.unsqueeze(1) * k).sum(dim=2)               # [B, K, H, W]
#         else:
#             logits = torch.einsum('b c h w, b k c h w -> b k h w', q, k) / self.scale

#         # 在 K（层）维度上 softmax，每个空间位置得到一组跨层权重
#         attn_weights = F.softmax(logits * self.tau.clamp(min=0.1, max=8.0), dim=1)
#         attn_out = (attn_weights.unsqueeze(2) * v).sum(dim=1)      # [B, C, H, W]

#         return x + self.out_proj(attn_out)

# 0821版本 n尺度微小提升，s尺度无提升，还下降了
# class AttnRes(nn.Module):
#     """
#     通道维度注意力残差模块：对跨层特征在通道维度做 softmax 激活，
#     实现自适应特征融合。无额外存储，轻量化。
#     """
#     def __init__(self, channels, use_cosine_sim=True):
#         super().__init__()
#         self.channels = channels
#         self.use_cosine_sim = use_cosine_sim

#         self.q_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.k_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.v_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
#         self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

#         # 数值稳定
#         self.eps = 1e-8

#         # 点积分支的缩放因子
#         self.scale = channels ** 0.5 if not use_cosine_sim else 1.0

#     def forward(self, x, history_feats):
#         """
#         x: 当前特征 [B, C, H, W]
#         history_feats: 对齐后的历史特征列表，每个 [B, C, H, W]
#         输出: 融合后的特征 [B, C, H, W]
#         """
#         feats = history_feats + [x]   # 所有参与融合的层
#         K = len(feats)

#         # Query: 当前特征
#         q = self.q_proj(x)            # [B, C, H, W]

#         # 收集各层的相似度得分和 value
#         logits = []                   # 每个元素 [B, 1, H, W]
#         values = []                   # 每个元素 [B, C, H, W]

#         for feat in feats:
#             k = self.k_proj(feat)     # [B, C, H, W]
#             v = self.v_proj(feat)     # [B, C, H, W]

#             if self.use_cosine_sim:
#                 # 余弦相似度：在通道维度上归一化后点积求和
#                 q_norm = F.normalize(q, p=2, dim=1, eps=self.eps)  # [B, C, H, W]
#                 k_norm = F.normalize(k, p=2, dim=1, eps=self.eps)  # [B, C, H, W]
#                 sim = (q_norm * k_norm).sum(dim=1, keepdim=True)    # [B, 1, H, W]
#             else:
#                 # 缩放点积相似度
#                 sim = (q * k).sum(dim=1, keepdim=True) / self.scale  # [B, 1, H, W]

#             logits.append(sim)
#             values.append(v)

#         # 将各层相似度拼接，在层维度 dim=1 上做 softmax
#         logits = torch.cat(logits, dim=1)          # [B, K, H, W]
#         attn_weights = F.softmax(logits, dim=1)    # [B, K, H, W]

#         # 加权聚合 value（注意：不需要再除以 K）
#         fused = torch.zeros_like(values[0])
#         for i in range(K):
#             fused = fused + attn_weights[:, i:i+1] * values[i]  # [B, C, H, W]

#         # 残差输出
#         out = x + self.out_proj(fused)
#         return out

    
    
class C2f_AttnRes(C2f):
    """
    集成通道维度AttnRes的C2f模块，修复重复定义self.m的问题。
    """
    def __init__(
        self, 
        c1, c2, n=1, shortcut=True, g=1, e=0.5, 
        stride=4, use_attnres=True, mode= 'P4', use_cosine_sim=True
    ):
        super().__init__(c1, c2, n, shortcut, g, e)   # 父类已创建self.m
        self.use_attnres = use_attnres
        self.hidden_c = int(c2 * e)
        self.stride = stride
        self.mode = mode

        # 特征对齐模块
        if use_attnres:
            self.align_module1 = nn.Sequential(
                Focus(c1 // self.stride, self.hidden_c, k=1),
                Conv(self.hidden_c, self.hidden_c, k=3, s=2, act=True)
            )

            self.align_module2 = nn.Sequential(
                Focus(c1 // 2, self.hidden_c, k=1),
            )
            # 初始化通道维度AttnRes
            self.attnres = AttnRes(
                channels=self.hidden_c,
                use_cosine_sim=use_cosine_sim
            )

    def forward(self, x, history_feats=None):
        y = list(self.cv1(x).chunk(2, 1))

        if self.use_attnres and history_feats is not None:
            history_aligned = []
            if self.mode == 'P5':
                history_aligned.append(self.align_module1(history_feats[0]))
                history_aligned.append(self.align_module2(history_feats[1]))
            else:
                history_aligned.append(self.align_module1(history_feats[0]))
            
            # 先做通道维度AttnRes，再送入Bottleneck
            y[-1] = self.attnres(y[-1], history_aligned)

        # 使用父类的Bottleneck列表（self.m）
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3k2_AttnRes(C3k2):
    """
    集成通道维度AttnRes的C3k2模块。
    """
    def __init__(
        self, 
        c1, c2, n=1, c3k=False, shortcut=True, g=1, e=0.5, 
        stride=4, use_attnres=True, mode='P4', use_cosine_sim=True
    ):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)   # 父类已创建self.m
        self.use_attnres = use_attnres
        self.hidden_c = int(c2 * e)
        self.stride = stride
        self.mode = mode

        # 特征对齐模块
        if use_attnres:
            self.align_module1 = nn.Sequential(
                Focus(c1 // self.stride, self.hidden_c, k=1),
                Conv(self.hidden_c, self.hidden_c, k=3, s=2, act=True)
            )

            self.align_module2 = nn.Sequential(
                Focus(c1 // 2, self.hidden_c, k=1),
            )
            # 初始化通道维度AttnRes
            self.attnres = AttnRes(
                channels=self.hidden_c,
                use_cosine_sim=use_cosine_sim
            )

    def forward(self, x, history_feats=None):
        y = list(self.cv1(x).chunk(2, 1))

        if self.use_attnres and history_feats is not None:
            history_aligned = []
            if self.mode == 'P5':
                history_aligned.append(self.align_module1(history_feats[0]))
                history_aligned.append(self.align_module2(history_feats[1]))
            else:
                history_aligned.append(self.align_module1(history_feats[0]))
            
            # 先做通道维度AttnRes，再送入Bottleneck
            y[-1] = self.attnres(y[-1], history_aligned)

        # 使用父类的Bottleneck列表（self.m）
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


if __name__ == '__main__':
    input = torch.randn(4, 64, 80, 80)  # 当前Neck层特征
    target = torch.randn(4, 128, 40, 40)  #
    model = C2f_AttnRes(c1=64, c2=128)
    output = model(input, target)
    print(output)