import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C3k2, C2f, Bottleneck, Focus, ChannelAttention

from einops import rearrange


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