import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.modules.block import C2f, Bottleneck


class AdaptiveWaveletPacketFusion(nn.Module):
    """
    自适应小波包融合模块 (AWPF)
    动机：模拟多级小波包分解，通过可学习权重自适应融合不同频率子带特征。
    设计：使用不同扩张率的深度可分离卷积来近似不同尺度的频率分析，这是一种轻量化且有效的设计[1]。
    """
    def __init__(self, in_channels, num_subbands=4):
        super().__init__()
        self.num_subbands = num_subbands
        # 使用不同扩张率的深度可分离卷积来提取多尺度（多子带）特征
        # 扩张率dilation增大，感受野增大，对应更低频的成分（近似小波包的低频子带）
        self.dw_convs = nn.ModuleList()
        for i in range(num_subbands):
            # 扩张率从1开始递增，模拟不同尺度
            dilation = 2 ** i
            # 使用5x5深度可分离卷积扩大感受野，与PP-PicoDet设计一致[1]
            self.dw_convs.append(
                nn.Sequential(
                    # Depthwise Conv
                    nn.Conv2d(in_channels, in_channels, kernel_size=5,
                              padding=2*dilation, dilation=dilation, groups=in_channels, bias=False),
                    nn.BatchNorm2d(in_channels),
                    nn.Hardswish(inplace=True),  # 使用H-Swish激活函数，对移动端友好且性能更优[11][30]
                )
            )
        # 可学习的子带融合权重
        self.fusion_weights = nn.Parameter(torch.ones(num_subbands) / num_subbands)
        # 融合后的1x1点卷积，用于通道混合和降维（如需）
        self.pw_conv = Conv(in_channels, in_channels, k=1)
    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: fused_feat [B, C, H, W]
        """
        subband_features = []
        for dw_conv in self.dw_convs:
            subband_features.append(dw_conv(x))
        # 自适应加权融合: 对权重进行softmax，确保和为1
        weights = F.softmax(self.fusion_weights, dim=0)
        weighted_sum = sum(w * f for w, f in zip(weights, subband_features))
        # 通过点卷积进行最终融合与调整
        fused_feat = self.pw_conv(weighted_sum)
        return fused_feat
    

# -------------------- 核心模块2：小波包多尺度解耦注意力 (WPD-SDSA) --------------------
class WPD_SDSA(nn.Module):
    """
    小波包多尺度解耦注意力模块 (Wavelet Packet Decoupled Self-Attention)
    这是方案二的核心，集成了特征解耦和注意力机制，摒弃了复杂的对抗学习。
    流程: 输入 -> 1x1投影 -> AWPF多尺度引导 -> 通道注意力 -> 特征增强 -> 残差输出
    """
    def __init__(self, in_channels, reduction_ratio=16):
        super().__init__()
        self.in_channels = in_channels
        # 1. 特征投影层: 将输入特征映射到隐空间
        self.proj = Conv(in_channels, in_channels, k=1)
        # 2. 自适应小波包融合模块 (AWPF): 生成多尺度引导特征
        self.awpf = AdaptiveWaveletPacketFusion(in_channels, num_subbands=4)
        # 3. 通道注意力机制 (轻量化): 基于多尺度引导特征产生注意力权重
        #    使用SE模块[8]的思想，但输入是我们的多尺度引导特征
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化，得到通道描述符
            Conv(in_channels, max(in_channels // reduction_ratio, 8), k=1), # 保证最小通道数
            nn.Hardswish(inplace=True),
            Conv(max(in_channels // reduction_ratio, 8), in_channels, k=1, act=False),
            nn.Sigmoid()  # 输出归一化的通道重要性权重
        )
        # 4. 输出投影层 (可选，用于调整通道或进一步融合)
        self.out_conv = Conv(in_channels, in_channels, k=1)
    def forward(self, x):
        """
        输入: x [B, C, H, W]
        输出: out [B, C, H, W]
        """
        identity = x  # 保留残差连接
        # 步骤A: 特征投影
        x_proj = self.proj(x)
        # 步骤B: 生成多尺度引导特征
        guided_feat = self.awpf(x_proj)
        # 步骤C: 计算通道注意力权重并应用于投影特征
        #        这里引导特征用于“解耦”注意力，使其更关注尺度不变信息
        attn_weights = self.channel_attention(guided_feat)
        x_enhanced = x_proj * attn_weights
        # 步骤D: 最终输出 (残差连接)
        out = self.out_conv(x_enhanced)
        out = out + identity
        return out
    

# -------------------- 模块集成1：WPD-SDSA Bottleneck --------------------
class Bottleneck_WPD_SDSA(Bottleneck):
    """
    集成了WPD-SDSA模块的Bottleneck。
    设计: 替换标准Bottleneck中的第二个3x3卷积为WPD-SDSA模块。
          这相当于在残差路径中加入了一个强大的特征增强单元。
    """
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        # 调用父类初始化，但我们会修改cv2
        super().__init__(c1, c2, shortcut, g, k, e)
        # 替换原始的cv2 (标准3x3 Conv) 为我们的WPD-SDSA模块
        # 注意：WPD-SDSA输入输出通道数相同，所以这里用c_（隐藏层通道数）
        self.sdsa = WPD_SDSA(self.c)
        # 由于WPD-SDSA内部已经包含了必要的非线性激活和归一化，
        # 我们可以简化或移除父类Bottleneck中的一些层（这里选择简化，仅保留核心结构）
        # 注意：实际替换时需要确保通道数匹配。这里假设self.c是中间通道数。
    def forward(self, x):
        """前向传播: 与标准Bottleneck类似，但用WPD-SDSA替换第二个卷积"""
        x_proj = self.cv1(x)  # 第一个1x1卷积
        # 应用WPD-SDSA进行特征增强
        x_enhanced = self.sdsa(x_proj)
        # 第二个1x1卷积 (如果shortcut为True，则与输入相加)
        return x_enhanced + x if self.shortcut else x_enhanced
        # 注意：这里是一个简化实现。严格来说，标准Bottleneck的cv2是1x1卷积。
        # 更合理的集成方式是：保持cv1和cv2为1x1卷积，在中间插入WPD-SDSA。
        # 下面提供一个更准确的版本：
class Bottleneck_WPD_SDSA_V2(nn.Module):
    """一个更清晰准确的WPD-SDSA Bottleneck实现"""
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)  # 隐藏通道数
        # 第一个1x1卷积，用于升维或降维
        self.cv1 = Conv(c1, c_, 1, 1)
        # 核心：WPD-SDSA模块，在隐藏维度上进行特征增强
        self.sdsa = WPD_SDSA(c_)
        # 第二个1x1卷积，将通道数调整回c2
        self.cv2 = Conv(c_, c2, 1, 1)
        self.shortcut = shortcut and c1 == c2
    def forward(self, x):
        identity = x
        out = self.cv1(x)
        out = self.sdsa(out)
        out = self.cv2(out)
        if self.shortcut:
            out = out + identity
        return out


# -------------------- 集成SDSA的C2f --------------------
class C2fSDSA(C2f):
    """
    集成了WPD-SDSA Bottleneck的C2f模块。
    策略：用WPD-SDSA Bottleneck替换一部分原始Bottleneck，形成混合C2f模块。
          这种设计平衡了性能提升和参数增加，是轻量化模型的常见做法。
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        # 先调用父类初始化，得到基础的C2f结构（包含m列表）
        super().__init__(c1, c2, n, shortcut, g, e)
        # 计算用WPD-SDSA Bottleneck替换的数量（例如替换一半）
        num_sdsa = n // 2
        # 重新构建m模块列表
        new_m = nn.ModuleList()
        for i in range(n):
            if i < num_sdsa:
                # 前一半使用WPD-SDSA Bottleneck
                new_m.append(Bottleneck_WPD_SDSA_V2(self.c, self.c, shortcut, g, e=1.0))
            else:
                # 后一半使用原始Bottleneck (从父类m中获取，或新建)
                # 注意：这里简化处理，直接新建原始Bottleneck。更严谨的做法是从父类m中提取。
                new_m.append(Bottleneck(self.c, self.c, shortcut, g, e=1.0))
        self.m = new_m  # 替换原有的m


# -------------------- 测试代码 --------------------
if __name__ == "__main__":
    # 1. 初始化模型
    channels = 256
    model = C2fSDSA(channels, 256, 2, shortcut=True)

    # 2. 模拟输入 (支持奇数尺寸)
    x = torch.randn(2, channels, 32, 32)  # Batch=2, C=256, H=32, W=32
    print(f"Input shape: {x.shape}")

    # 3. 前向传播
    output = model(x)
    print(f"Output shape: {output.shape}")

    # 4. 验证尺寸一致性
    assert x.shape == output.shape, "输入输出尺寸不一致！"
    print("✅ 测试通过：小波增强模块运行正常，尺寸一致。")