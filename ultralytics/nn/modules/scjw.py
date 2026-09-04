import torch
import torch.nn as nn
import torch.nn.functional as F

class SCJW(nn.Module):
    def __init__(self, c1: int, reduction=4):
        """
        初始化模块
        :param dim: 输入特征图的通道数C，对应公式里的C
        :param reduction: 通道压缩比例，用来减少计算量，默认4就行
        """
        super().__init__()
        # 通道注意力：类似SE，但输入为前景-背景差异+全局池化
        self.fc_c = nn.Sequential(
            nn.Conv2d(c1, c1 // reduction, 1),
            nn.SiLU(),
            nn.Conv2d(c1 // reduction, c1, 1),
            nn.Sigmoid()
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, box_mask=None):
        B, C, H, W = x.shape

        # ------------------- 如果没有mask，直接返回原特征（或经过增强） -------------------
        if box_mask is None:
            return x   # 你可以选择是否经过增强层

        # ------------------- 1. 预处理mask -------------------
        mask_down = F.interpolate(box_mask, size=(H, W), mode='nearest')  # [B,1,H,W]
        wh = H * W

        # 计算前景和背景像素数，并处理无前景的情况
        fore_pix = mask_down.sum(dim=[2,3], keepdim=True)                 # [B,1,1,1]
        back_pix = wh - fore_pix
        # 如果前景像素太少（例如小于10），则放弃使用注意力（r_c=1, r_s=1）
        fore_valid = fore_pix > 10   # 阈值可调，避免噪声影响
        # 但为了简便，这里仍用 clamp 防止除零，同时后续对无效样本特殊处理
        fore_pix = torch.clamp(fore_pix, min=1)
        back_pix = torch.clamp(back_pix, min=1)

        # ------------------- 2. 计算前景/背景平均特征 -------------------
        fore_feat = (x * mask_down).sum(dim=[2,3], keepdim=True) / fore_pix   # [B,C,1,1]
        back_feat = (x * (1 - mask_down)).sum(dim=[2,3], keepdim=True) / back_pix

        # ------------------- 3. 通道权重（改进） -------------------
        phi = fore_feat - back_feat                     # 前景-背景差异，[B,C,1,1]

        # 方法A：对差异做标准化 + sigmoid，保证权重在(0,1)之间
        # phi_mean = phi.mean(dim=1, keepdim=True)        # 对每个样本的通道求均值
        # phi_std = phi.std(dim=1, keepdim=True).clamp(min=1e-5)
        # phi_norm = (phi - phi_mean) / phi_std           # 标准化后均值为0，标准差1
        # r_c = torch.sigmoid(phi_norm)                   # [B,C,1,1]，每个通道独立权重

        # 方法B：直接用差异通过一个小网络生成权重（类似SE）
        gap = x.mean(dim=[2,3], keepdim=True)
        feat_c = phi + gap  # 简单融合
        r_c = self.fc_c(feat_c)  # [B,C,1,1]

        # 如果你希望权重之和为1（竞争性注意力），可以用 softmax：
        # r_c = F.softmax(phi_norm, dim=1)               # 但这样可能会使某些通道完全被抑制

        # ------------------- 4. 空间权重（改进） -------------------
        # 前景原型仍用 fore_feat
        f_s = fore_feat                                  # [B,C,1,1]
        # 计算余弦相似度（归一化后点乘）
        x_norm = F.normalize(x, p=2, dim=1)              # 对特征做通道L2归一化
        f_s_norm = F.normalize(f_s, p=2, dim=1)
        sim = (x_norm * f_s_norm).sum(dim=1, keepdim=True)  # [B,1,H,W]，范围[-1,1]

        # 用 sigmoid 将相似度映射到(0,1)，并增加一个缩放因子突出对比度
        scale = 5.0                                        # 可调，越大越接近0/1二值
        r_s = torch.sigmoid(scale * sim)                  # [B,1,H,W]

        # ------------------- 5. 处理无前景的样本 -------------------
        # 如果某个样本的前景像素数太少（fore_valid 为 False），我们可以强制其 r_c=1, r_s=1
        # 但为了简单，这里暂时不做，因为 clamp 后 fore_pix=1 也会得到一个弱的前景特征，影响不大
        # 如果有需要，可以这样：
        # r_c = r_c * fore_valid.float() + (~fore_valid).float() * 1.0   # 无效样本权重设为1
        # r_s = r_s * fore_valid.float() + (~fore_valid).float() * 1.0

        # ------------------- 6. 联合加权 -------------------
        weighted_x = x * r_c * r_s

        # with torch.no_grad():
        #         print(f"[SCJW] r_c mean: {r_c.mean().item():.4f}, std: {r_c.std().item():.4f}, "
        #               f"min: {r_c.min().item():.4f}, max: {r_c.max().item():.4f}")
        #         print(f"[SCJW] r_s mean: {r_s.mean().item():.4f}, std: {r_s.std().item():.4f}, "
        #               f"min: {r_s.min().item():.4f}, max: {r_s.max().item():.4f}")

        # ------------------- 7. 特征增强 -------------------
        # enhanced_x = self.feature_enhance(weighted_x)

        return weighted_x
