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
    def __init__(self, in_channels=1, num_outputs=8, use_bn=False, separate_heads=False, input_hw=(60, 60)):
        super().__init__()
        self.stem = Conv2dBNReLU(in_channels, 96, 3, p=1, use_bn=use_bn)
        self.A    = nn.Sequential(*[InceptionARes(use_bn=use_bn) for _ in range(4)])
        self.RedA = ReductionARes(use_bn=use_bn)
        self.B    = nn.Sequential(*[InceptionBRes(use_bn=use_bn) for _ in range(10)])

        self.separate_heads = separate_heads

        # Infer flatten feature size once (Keras-like: it knows input shape; we emulate that)
        H, W = input_hw
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)   # on CPU; module is on CPU here
            x = self.B(self.RedA(self.A(self.stem(dummy))))
            in_feats = x.flatten(1).shape[1]            # e.g., 242208 for 60x60

        if not separate_heads:
            self.head = nn.Linear(in_feats, num_outputs)
        else:
            self.heads = nn.ModuleList([nn.Linear(in_feats, 1) for _ in range(num_outputs)])

    def forward(self, x):
        x = self.stem(x)
        x = self.A(x)
        x = self.RedA(x)
        x = self.B(x)

        x = x.flatten(1)  # FULL spatial flatten (paper-style)

        if hasattr(self, "head"):
            return self.head(x)                         # (B, num_outputs)
        outs = [h(x) for h in self.heads]              # list of (B,1)
        return torch.cat(outs, dim=1)                  # (B, num_outputs)

class SoftDwellThenTFIR(nn.Module):
    """
    Expects raw time series x: [B, C, T]
    softdwell produces H: [B, K, H, W]
    then feeds H to TFExactInceptionResNet.
    """
    def __init__(
        self,
        softdwell_layer: nn.Module,
        K: int,                       # number of histogram channels output by softdwell
        hist_hw=(60, 60),             # histogram size
        num_outputs=8,
        use_bn=False,
        separate_heads=False,
    ):
        super().__init__()
        self.softdwell = softdwell_layer

        self.backbone = TFExactInceptionResNet(
            in_channels=K,
            num_outputs=num_outputs,
            use_bn=use_bn,
            separate_heads=separate_heads,
            input_hw=hist_hw,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logH = self.softdwell(x)   # [B, K, H, W]
        return self.backbone(logH)
    