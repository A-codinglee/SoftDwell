import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k, s=1, p=0, use_bn=False):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_ch) if use_bn else None
    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class InceptionARes(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        C = 96
        self.b1_1 = Conv2dBNReLU(C, 8, 1, use_bn=use_bn)
        self.b2_1 = Conv2dBNReLU(C, 8, 1, use_bn=use_bn)
        self.b2_2 = Conv2dBNReLU(8, 8, 3, p=1, use_bn=use_bn)
        self.b3_1 = Conv2dBNReLU(C, 8, 1, use_bn=use_bn)
        self.b3_2 = Conv2dBNReLU(8, 12, 3, p=1, use_bn=use_bn)
        self.b3_3 = Conv2dBNReLU(12, 16, 3, p=1, use_bn=use_bn)
        self.mix = nn.Conv2d(8 + 8 + 16, 96, 1, bias=not use_bn)
        self.mix_bn = nn.BatchNorm2d(96) if use_bn else None
        self.out_bn = nn.BatchNorm2d(96) if use_bn else None
    def forward(self, x):
        x1 = self.b1_1(x)
        x2 = self.b2_2(self.b2_1(x))
        x3 = self.b3_3(self.b3_2(self.b3_1(x)))
        xc = torch.cat([x1, x2, x3], dim=1)
        xc = self.mix(xc)
        if self.mix_bn is not None:
            xc = self.mix_bn(xc)
        out = x + xc
        if self.out_bn is not None:
            out = self.out_bn(out)
        return F.relu(out, inplace=True)

class ReductionARes(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        C = 96
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.b2 = Conv2dBNReLU(C, 96, 3, s=2, use_bn=use_bn)
        self.b3_1 = Conv2dBNReLU(C, 64, 1, use_bn=use_bn)
        self.b3_2 = Conv2dBNReLU(64, 64, 3, p=1, use_bn=use_bn)
        self.b3_3 = Conv2dBNReLU(64, 96, 3, s=2, use_bn=use_bn)
    def forward(self, x):
        y1 = self.pool(x)                   # (N,96,29,29)
        y2 = self.b2(x)                     # (N,96,29,29)
        y3 = self.b3_3(self.b3_2(self.b3_1(x)))  # (N,96,29,29)
        return torch.cat([y1, y2, y3], dim=1)    # (N,288,29,29)

class InceptionBRes(nn.Module):
    def __init__(self, use_bn=False):
        super().__init__()
        C = 288
        self.b1_1 = Conv2dBNReLU(C, 48, 1, use_bn=use_bn)
        self.b2_1 = Conv2dBNReLU(C, 32, 1, use_bn=use_bn)
        self.b2_2 = Conv2dBNReLU(32, 40, (1, 7), p=(0, 3), use_bn=use_bn)
        self.b2_3 = Conv2dBNReLU(40, 48, (7, 1), p=(3, 0), use_bn=use_bn)
        self.mix = nn.Conv2d(48 + 48, 288, 1, bias=not use_bn)
        self.mix_bn = nn.BatchNorm2d(288) if use_bn else None
        self.out_bn = nn.BatchNorm2d(288) if use_bn else None
    def forward(self, x):
        x1 = self.b1_1(x)
        x2 = self.b2_3(self.b2_2(self.b2_1(x)))
        xc = torch.cat([x1, x2], dim=1)
        xc = self.mix(xc)
        if self.mix_bn is not None:
            xc = self.mix_bn(xc)
        out = x + xc
        if self.out_bn is not None:
            out = self.out_bn(out)
        return F.relu(out, inplace=True)

class TFExactInceptionResNet(nn.Module):
    def __init__(self, in_channels=1, num_outputs=8, use_bn=False, separate_heads=False):
        super().__init__()
        self.stem = Conv2dBNReLU(in_channels, 96, 3, p=1, use_bn=use_bn)
        self.A   = nn.Sequential(*[InceptionARes(use_bn=use_bn) for _ in range(4)])
        self.RedA= ReductionARes(use_bn=use_bn)
        self.B   = nn.Sequential(*[InceptionBRes(use_bn=use_bn) for _ in range(10)])
        self.pool= nn.AdaptiveAvgPool2d((1, 1))
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(288, 1) for _ in range(num_outputs)])
        else:
            self.head  = nn.Linear(288, num_outputs)
    def forward(self, x):
        x = self.stem(x)         # (N,96,60,60)
        x = self.A(x)            # (N,96,60,60)
        x = self.RedA(x)         # (N,288,29,29)
        x = self.B(x)            # (N,288,29,29)
        x = self.pool(x).flatten(1)  # (N,288)
        if self.separate_heads:
            outs = [h(x) for h in self.heads]
            return torch.cat(outs, dim=1)
        return self.head(x)
    

class HOHDThenTFIR(nn.Module):
    """
    Expects raw time series x: [B, C, T]
    HOHD produces hist H: [B, K, 60, 60] where K = C * D (D = num_detectors).
    Then feeds H to your TFExactInceptionResNet.

    compress:
      - 'none'        : use all K channels (set backbone in_channels = K)
      - 'mean_det'    : average over detectors → in_channels = C
      - '1x1'         : learnable 1x1 projection K→1 → in_channels = 1
    """
    def __init__(self, hohd_layer: nn.Module, num_outputs=8, use_bn=True,
                 separate_heads=False, compress: str = "none", use_log1p=True):
        super().__init__()
        self.hohd = hohd_layer
        self.use_log1p = use_log1p
        self.compress = compress
        self.use_bn = use_bn
        self.num_outputs = num_outputs
        self.separate_heads = separate_heads

        self.backbone = None     # lazy-build after we know K
        self.chan_proj = None    # for compress='1x1'

    def _build_backbone(self, K: int, device):
        # Decide input channels for your backbone
        if self.compress == "mean_det":
            D = self.hohd.D
            C = K // D
            in_ch = C
        elif self.compress == "1x1":
            in_ch = 1
            self.chan_proj = Conv2dBNReLU(K, 1, 1, use_bn=self.use_bn)
        else:
            in_ch = K

        self.backbone = TFExactInceptionResNet(
            in_channels=in_ch,
            num_outputs=self.num_outputs,
            use_bn=self.use_bn,
            separate_heads=self.separate_heads,
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        H = self.hohd(x)                    # [B, K, 60, 60]

        if self.backbone is None:           # lazy init once K is known
            self._build_backbone(H.shape[1], H.device)

        if self.use_log1p:
            H = torch.log1p(H)

        if self.compress == "mean_det":
            B, K, Hh, Ww = H.shape
            D = self.hohd.D
            C = K // D
            H = H.view(B, C, D, Hh, Ww).mean(dim=2)   # [B, C, 60, 60]
        elif self.compress == "1x1":
            H = self.chan_proj(H)                     # [B, 1, 60, 60]

        return self.backbone(H)