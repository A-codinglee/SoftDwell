import functools
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
    def __init__(self, in_channels=1, num_outputs=8, use_bn=False, separate_heads=False, input_hw=(60, 60), no_head=False):
        super().__init__()
        self.stem = Conv2dBNReLU(in_channels, 96, 3, p=1, use_bn=use_bn)
        self.A    = nn.Sequential(*[InceptionARes(use_bn=use_bn) for _ in range(4)])
        self.RedA = ReductionARes(use_bn=use_bn)
        self.B    = nn.Sequential(*[InceptionBRes(use_bn=use_bn) for _ in range(10)])

        self.separate_heads = separate_heads
        self.no_head = no_head

        # Infer flatten feature size once (Keras-like: it knows input shape; we emulate that)
        H, W = input_hw
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, H, W)   # on CPU; module is on CPU here
            x = self.B(self.RedA(self.A(self.stem(dummy))))
            in_feats = x.flatten(1).shape[1]            # e.g., 242208 for 60x60
        self.out_features = in_feats                    # exposed for late-fusion wrappers (see HybridLateFusionNet)

        if no_head:
            pass                                        # forward() returns the flattened features directly
        elif not separate_heads:
            self.head = nn.Linear(in_feats, num_outputs)
        else:
            self.heads = nn.ModuleList([nn.Linear(in_feats, 1) for _ in range(num_outputs)])

    def forward(self, x):
        x = self.stem(x)
        x = self.A(x)
        x = self.RedA(x)
        x = self.B(x)

        x = x.flatten(1)  # FULL spatial flatten (paper-style)

        if self.no_head:
            return x                                    # (B, out_features)
        if hasattr(self, "head"):
            return self.head(x)                         # (B, num_outputs)
        outs = [h(x) for h in self.heads]              # list of (B,1)
        return torch.cat(outs, dim=1)                  # (B, num_outputs)

class SmallCNNBackbone(nn.Module):
    """Right-sized alternative to TFExactInceptionResNet for the tiny (J~12-14) histogram inputs this
    pipeline actually uses. TFExactInceptionResNet is a near-literal port of Inception-ResNet-v2's early
    stages (4x InceptionA-Res -> one stride-2 reduction -> 10x InceptionB-Res), built for 299x299 ImageNet
    images; traced on our actual shapes, the one reduction collapses a 12x12/14x14 input to 5x5/6x6 within
    the first few layers, and then runs 10 MORE multi-branch Inception blocks on that already-tiny grid --
    most of Inception's whole design premise (parallel branches at different kernel scales) is close to
    meaningless once the grid is 5x5, and the full-flatten head (not global pooling) turns that into a
    large parameter count for datasets of only 10-40k rows.

    This backbone instead: plain 3x3 conv stack, NO spatial downsampling (stride 1 throughout -- the input
    is already tiny, nothing to gain from shrinking it further), small constant channel width, global
    average pooling instead of a full spatial flatten (so out_features is fixed regardless of input_hw,
    unlike TFExactInceptionResNet which needs a dummy forward pass to infer it). Same constructor/forward
    interface (in_channels, num_outputs, use_bn, separate_heads, input_hw accepted but unused, no_head) so
    it drops into any fusion architecture in place of TFExactInceptionResNet with no other code changes.
    """
    def __init__(self, in_channels=1, num_outputs=8, use_bn=True, separate_heads=False, input_hw=(12, 12),
                 no_head=False, hidden=64, depth=4):
        super().__init__()
        layers = [Conv2dBNReLU(in_channels, hidden, 3, p=1, use_bn=use_bn)]
        for _ in range(depth - 1):
            layers.append(Conv2dBNReLU(hidden, hidden, 3, p=1, use_bn=use_bn))
        self.convs = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features = hidden   # fixed regardless of input_hw -- no dummy-forward shape inference needed
        self.no_head = no_head
        self.separate_heads = separate_heads
        if no_head:
            pass
        elif not separate_heads:
            self.head = nn.Linear(hidden, num_outputs)
        else:
            self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_outputs)])

    def forward(self, x):
        x = self.convs(x)
        x = self.pool(x).flatten(1)          # (B, hidden)
        if self.no_head:
            return x
        if hasattr(self, "head"):
            return self.head(x)
        return torch.cat([h(x) for h in self.heads], dim=1)


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2). Operates on channels-last (B,H,W,C) tensors -- the
    block below always calls this right after its (B,C,H,W)->(B,H,W,C) permute for the LayerNorm/Linear
    steps, same as every other op inside the block. gamma/beta learnable, zero-initialized, so this starts
    as an exact identity (x -> x) and only departs from it as training finds it useful -- same "don't
    disturb the starting point" convention as every other learned-offset component in this codebase.
    ConvNeXt V2's replacement for V1's LayerScale: instead of a plain per-channel learned scale on the
    residual branch, GRN computes each channel's L2 response magnitude across space, normalizes it against
    the mean response across ALL channels, and uses that to rescale the features -- so channels compete:
    a channel that's unusually active relative to its peers (for this particular input) gets amplified,
    a channel that's flat gets suppressed. This is the actual ConvNeXt V2 contribution (paired in the
    original paper with FCMAE self-supervised pretraining, which doesn't apply here -- we use only the
    block-design change, training from scratch like every other backbone in this file).
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):   # x: (B,H,W,C)
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)        # (B,1,1,C) -- per-channel spatial L2 norm
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)          # normalize against the cross-channel mean
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """One ConvNeXt V2 block: depthwise large-kernel spatial mixing (the "token mixer", analogous to
    attention in a transformer block) -> LayerNorm -> pointwise inverted-bottleneck MLP (the "channel
    mixer", analogous to a transformer's FFN: expand mlp_ratio x, GELU, project back) -> GRN -> residual
    add. Kernel size is NOT the usual ConvNeXt 7 by default here -- our spatial extent is only J~12-14, so
    a 7x7 depthwise conv (padding=3, same spatial size preserved) already reaches nearly the whole grid in
    a single layer; this is a deliberate consequence of the "don't downsample a 12x12 input further"
    lesson from SmallCNNBackbone/SmallCNN3DBackbone, not an oversight.
    """
    def __init__(self, dim, mlp_ratio=4, kernel_size=7, use_grn=True):
        super().__init__()
        pad = kernel_size // 2
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=pad, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.grn = GRN(mlp_ratio * dim) if use_grn else None   # ablation switch: use_grn=False isolates
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)         # GRN's own contribution (model.py's
                                                                 # ConvNeXtV2Backbone(use_grn=False) wires this)

    def forward(self, x):   # x: (B,C,H,W)
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)          # (B,H,W,C) -- LayerNorm/Linear/GRN all operate channels-last
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        if self.grn is not None:
            x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)          # back to (B,C,H,W)
        return inp + x


class ConvNeXtV2Backbone(nn.Module):
    """Right-sized ConvNeXt V2 for our tiny (J~12-14) histogram inputs -- NOT a literal port of the
    original (which patchifies a 224x224+ image with a strided 4x4 stem, then downsamples 4 more times
    across stages). Both of those moves would be destructive here (a 4x4-stride stem alone would collapse
    a 12x12 input to 3x3 before a single ConvNeXt block even runs), so: a plain 1x1 conv stem (projects
    the K detector channels to `hidden` width, no spatial change at all) followed by a flat stack of
    ConvNeXtV2Block at constant spatial resolution and constant channel width -- same "no unnecessary
    downsampling, small constant width" principle as SmallCNNBackbone, but using ConvNeXt V2's actual
    block design (depthwise-conv token mixer + GRN'd inverted-bottleneck channel MLP + LayerNorm/GELU)
    instead of plain Conv2d+BN+ReLU.
    """
    def __init__(self, in_channels=1, num_outputs=8, use_bn=True, separate_heads=False, input_hw=(12, 12),
                 no_head=False, hidden=64, depth=4, kernel_size=7, mlp_ratio=4, use_grn=True):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, hidden, kernel_size=1)   # channel projection only, no spatial change
        self.blocks = nn.Sequential(*[ConvNeXtV2Block(hidden, mlp_ratio=mlp_ratio, kernel_size=kernel_size, use_grn=use_grn)
                                       for _ in range(depth)])
        self.final_norm = nn.LayerNorm(hidden, eps=1e-6)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.out_features = hidden
        self.no_head = no_head
        self.separate_heads = separate_heads
        if no_head:
            pass
        elif not separate_heads:
            self.head = nn.Linear(hidden, num_outputs)
        else:
            self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_outputs)])

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)        # (B, hidden)
        x = self.final_norm(x)
        if self.no_head:
            return x
        if hasattr(self, "head"):
            return self.head(x)
        return torch.cat([h(x) for h in self.heads], dim=1)


class SmallCNN3DBackbone(nn.Module):
    """K-aware variant of SmallCNNBackbone. The histogram tensor is (B, K, J, J) -- K detectors, each a
    J-by-J dwell-time histogram. Every backbone elsewhere in this codebase (TFExactInceptionResNet,
    SmallCNNBackbone) treats K as an ordinary "channel" axis: a 2D conv's first layer freely mixes ALL K
    channels together at every spatial location, exactly as it would for RGB channels, with no notion that
    channel order means anything.

    But K is NOT arbitrary here -- each detector index is a specific threshold POSITION along an ordered
    ladder (detector k and k+1 are adjacent thresholds, physically). A 2D conv discards that ordering at
    the very first layer. This backbone instead treats the histogram as a genuine 3D volume (K, J, J) and
    uses Conv3d with a real kernel extent along K too (default 3), so the network can learn local structure
    ALONG the threshold ladder the same way it already does across the two dwell-time axes, instead of
    collapsing K into unordered "channels" immediately.

    Input reshaped (B, K, J, J) -> (B, 1, K, J, J) (a single input "channel", K/J/J as the 3 volumetric
    dims) before the first Conv3d; num_detectors_in (not in_channels) says how many physical input
    channels the RAW trace/histogram source itself has (e.g. C=1 for a single-electrode recording) -- for
    every architecture in this codebase that's always 1, so this is simplified to assume exactly that.

    Same drop-in interface as SmallCNNBackbone (in_channels here means K, kept for call-site compatibility
    with TFExactInceptionResNet/SmallCNNBackbone's constructor signature even though it means something
    conceptually different -- a volumetric depth, not a channel count).
    """
    def __init__(self, in_channels=1, num_outputs=8, use_bn=True, separate_heads=False, input_hw=(12, 12),
                 no_head=False, hidden=64, depth=4, k_kernel=3):
        super().__init__()
        k_pad = k_kernel // 2
        def conv3d_bn_relu(c_in, c_out):
            layers = [nn.Conv3d(c_in, c_out, kernel_size=(k_kernel, 3, 3), padding=(k_pad, 1, 1), bias=not use_bn)]
            if use_bn:
                layers.append(nn.BatchNorm3d(c_out))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)
        blocks = [conv3d_bn_relu(1, hidden)]
        for _ in range(depth - 1):
            blocks.append(conv3d_bn_relu(hidden, hidden))
        self.convs = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_features = hidden
        self.no_head = no_head
        self.separate_heads = separate_heads
        if no_head:
            pass
        elif not separate_heads:
            self.head = nn.Linear(hidden, num_outputs)
        else:
            self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(num_outputs)])

    def forward(self, x):
        # x: (B, K, J, J) -> (B, 1, K, J, J) -- K becomes a volumetric spatial dim, not a channel dim
        x = x.unsqueeze(1)
        x = self.convs(x)
        x = self.pool(x).flatten(1)          # (B, hidden)
        if self.no_head:
            return x
        if hasattr(self, "head"):
            return self.head(x)
        return torch.cat([h(x) for h in self.heads], dim=1)


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

    def forward(self, x: torch.Tensor = None, hard_fixed_hist=None, replace_soft_with=None) -> torch.Tensor:
        """
        replace_soft_with: (freeze/refresh experiment) a precomputed (B,K,H,W)
        tensor to use AS logH directly, skipping self.softdwell(x) entirely --
        no gradient path back to theta/tau (it's a plain cached tensor, no
        grad_fn), so this is how "frozen" epochs avoid both the O(T) compute
        AND any theta/tau update. Mutually exclusive with hard_fixed_hist
        (that's the separate hybrid_hard_fixed concatenated-branch feature).

        x is optional (default None) specifically so frozen-epoch callers can
        skip loading/transferring the raw (B,1,T) signal entirely -- it's
        never touched when replace_soft_with is given, so there is no reason
        to pay for it.
        """
        if replace_soft_with is not None:
            logH = replace_soft_with
        else:
            assert x is not None, "x is required unless replace_soft_with is given"
            logH = self.softdwell(x, hard_fixed_hist=hard_fixed_hist)   # [B, K, H, W]
        return self.backbone(logH)


class HybridLateFusionNet(nn.Module):
    """
    Late fusion of the two Family D single-method networks: each branch keeps its OWN backbone
    (no shared weights, no shared input channels -- this is the point, vs. hybrid_hard_fixed's early
    channel-concat which fuses before either representation has been processed at all). Each backbone
    runs to its flattened feature vector (no_head=True), the two vectors are concatenated, and ONE
    Linear(feat_hard + feat_soft, num_outputs) makes the final 9 predictions -- so each of the 9 outputs
    gets its own learned weighting over BOTH branches' features, rather than us hand-assigning
    "rates+sigma from hard, baseline+amplitude from soft" the way the zero-training ensemble does.

    hard branch:  precomputed (B, K_hard, J_hard, J_hard) fixed-ladder histogram (log10, no grad needed
                  into it -- the ladder is fixed, so this input is just data, like any other tensor).
    soft branch:  raw trace (B, 1, T) -> ChannelWiseSoftDwell (learned theta/tau) -> (B, K_soft, J_soft, J_soft).

    separate_heads=True: instead of one shared Linear(feat, num_outputs) -- which forces every output through
    the SAME joint weighting of the two branches' features -- each output gets its OWN Linear(feat, 1), so
    e.g. sigma is free to lean on the hard branch's features while amplitude leans on the soft branch's,
    independently. Motivated by the first fusion run: sigma correctly matched hard-threshold's accuracy, but
    amplitude landed worse than either single branch, consistent with one shared head compromising across
    labels that want different mixtures rather than a genuine failure of the represented information.
    """
    def __init__(self, softdwell_layer: nn.Module, k_hard: int, j_hard: int, k_soft: int, j_soft: int,
                 num_outputs: int = 9, use_bn: bool = False, separate_heads: bool = False):
        super().__init__()
        self.softdwell = softdwell_layer
        self.hard_backbone = TFExactInceptionResNet(in_channels=k_hard, use_bn=use_bn, input_hw=(j_hard, j_hard), no_head=True)
        self.soft_backbone = TFExactInceptionResNet(in_channels=k_soft, use_bn=use_bn, input_hw=(j_soft, j_soft), no_head=True)
        feat = self.hard_backbone.out_features + self.soft_backbone.out_features
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in range(num_outputs)])
        else:
            self.head = nn.Linear(feat, num_outputs)

    def forward(self, hard_hist: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """hard_hist: (B, K_hard, J_hard, J_hard) log10 counts, precomputed.  x_raw: (B, 1, T) raw trace, pA."""
        f_hard = self.hard_backbone(hard_hist)
        logH = self.softdwell(x_raw)                          # (B, K_soft, J_soft, J_soft)
        f_soft = self.soft_backbone(logH)
        f = torch.cat([f_hard, f_soft], dim=1)
        if self.separate_heads:
            return torch.cat([h(f) for h in self.heads], dim=1)
        return self.head(f)


class SelfCalibratingFusionNet(nn.Module):
    """
    Like HybridLateFusionNet, but the hard branch's histogram is no longer built from one FIXED global
    ladder -- it's built PER ROW from that row's own estimated baseline/amplitude, so the hard thresholds
    always straddle the row's actual open/closed levels instead of assuming a fixed absolute pA window.
    Motivation: Family D randomizes baseline (+-1 pA) and amplitude (2-20 pA) independently per row, so
    no fixed ladder -- uniform OR informed-but-fixed (see the learned-theta custom-threshold experiment)
    -- can bracket every row's levels at once. A per-row ladder can, by construction.

    Two-stage forward because the hard histogram has to be built (on the CPU/GPU, via grind_hist_gpu,
    OUTSIDE autograd -- it's a hard step function, same as every other histogram in this pipeline) from a
    level estimate that itself needs to come out of the soft branch first:

      1. soft_stage(x_raw)   -> f_soft, baseline_hat_pA, amp_hat_pA   (differentiable; compile-friendly)
      2. [caller builds the per-row hard histogram from baseline_hat_pA.detach(), amp_hat_pA.detach(),
         via grind_hist_gpu -- not part of this module, needs cfg/lut/counts_per_pA from the training
         script, same as every other hard-threshold histogram build in this codebase]
      3. finish(f_soft, hard_hist, baseline_hat_pA, amp_hat_pA) -> 9 outputs  (differentiable; compile-friendly)

    level_head gets gradient from TWO places now: the small AUXILIARY loss the training script adds on
    (baseline_hat_pA, amp_hat_pA) vs the true (baseline_pA, amplitude_pA), AND the main 9-output loss --
    because step 3 explicitly concatenates baseline_hat_pA/amp_hat_pA (NOT detached -- only the copy used
    to build the histogram in step 2 is detached) into the head's input. This closes a real gap the
    earlier version had: the head only ever saw f_hard (built from a per-row-recalibrated, hence
    position-erased, histogram) and f_soft -- it had no EXPLICIT signal for "what absolute window was
    this row's hard histogram actually centered on", only an indirect, unguaranteed path through f_soft
    (level_head is a linear readout of f_soft, so f_soft contains at least that information, but nothing
    forced the head to recover and use it). Explicitly feeding baseline_hat_pA/amp_hat_pA removes that
    gap directly, and also lets the main task's much stronger gradient refine the level estimate itself,
    not just the small auxiliary loss.

    baseline_hat_pA = tanh(raw[:,0]) * base_scale_pA      -- keeps early-training values in a sane range
    amp_hat_pA      = amp_min_pA + softplus(raw[:,1])     -- keeps amplitude positive from step 0
    """
    def __init__(self, softdwell_layer: nn.Module, k_hard: int, j_hard: int, k_soft: int, j_soft: int,
                 num_outputs: int = 9, use_bn: bool = False, separate_heads: bool = False,
                 base_scale_pA: float = 2.0, amp_min_pA: float = 1.0):
        super().__init__()
        self.softdwell = softdwell_layer
        self.hard_backbone = TFExactInceptionResNet(in_channels=k_hard, use_bn=use_bn, input_hw=(j_hard, j_hard), no_head=True)
        self.soft_backbone = TFExactInceptionResNet(in_channels=k_soft, use_bn=use_bn, input_hw=(j_soft, j_soft), no_head=True)
        self.level_head = nn.Linear(self.soft_backbone.out_features, 2)
        self.base_scale_pA = base_scale_pA
        self.amp_min_pA = amp_min_pA
        # +2 for the explicitly-fed baseline_hat_pA/amp_hat_pA scalars (see class docstring)
        feat = self.hard_backbone.out_features + self.soft_backbone.out_features + 2
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in range(num_outputs)])
        else:
            self.head = nn.Linear(feat, num_outputs)

    def soft_stage(self, x_raw: torch.Tensor):
        """x_raw: (B,1,T) pA. Returns (f_soft, baseline_hat_pA, amp_hat_pA), all (B,) or (B,feat)."""
        logH = self.softdwell(x_raw)                          # (B, K_soft, J_soft, J_soft)
        f_soft = self.soft_backbone(logH)
        raw = self.level_head(f_soft)                          # (B, 2)
        baseline_hat_pA = torch.tanh(raw[:, 0]) * self.base_scale_pA
        amp_hat_pA = self.amp_min_pA + F.softplus(raw[:, 1])
        return f_soft, baseline_hat_pA, amp_hat_pA

    def finish(self, f_soft: torch.Tensor, hard_hist: torch.Tensor,
               baseline_hat_pA: torch.Tensor, amp_hat_pA: torch.Tensor) -> torch.Tensor:
        """hard_hist: (B, K_hard, J_hard, J_hard) log10 counts, built per-row by the caller from a
        DETACHED copy of (baseline_hat_pA, amp_hat_pA). Here, pass the ORIGINAL (non-detached) tensors
        from soft_stage -- explicitly telling the head what absolute window that row's hard_hist was
        centered on, instead of leaving it implicit in f_soft (see class docstring)."""
        f_hard = self.hard_backbone(hard_hist)
        level = torch.stack([baseline_hat_pA, amp_hat_pA], dim=1)   # (B,2)
        f = torch.cat([f_hard, f_soft, level], dim=1)
        if self.separate_heads:
            return torch.cat([h(f) for h in self.heads], dim=1)
        return self.head(f)


class SelfCalibratingFusionNetV1(nn.Module):
    """Reconstruction of the ORIGINAL (pre-fix) SelfCalibratingFusionNet, kept only so the "v1" baseline
    can be re-trained under LOO for a matched comparison against v2 (SelfCalibratingFusionNet, above) --
    this repo is not under version control, and the v1->v2 fix was applied by editing the class in place,
    so the only other record of v1's exact behavior is its docstring history and training logs.

    Identical to SelfCalibratingFusionNet except finish() does NOT explicitly concatenate the level
    estimate into the head's input -- the head only ever sees f_hard and f_soft, same gap described in
    v2's class docstring. Do not use this for new experiments; it exists purely as the "before" arm.
    """
    def __init__(self, softdwell_layer: nn.Module, k_hard: int, j_hard: int, k_soft: int, j_soft: int,
                 num_outputs: int = 9, use_bn: bool = False, separate_heads: bool = False,
                 base_scale_pA: float = 2.0, amp_min_pA: float = 1.0):
        super().__init__()
        self.softdwell = softdwell_layer
        self.hard_backbone = TFExactInceptionResNet(in_channels=k_hard, use_bn=use_bn, input_hw=(j_hard, j_hard), no_head=True)
        self.soft_backbone = TFExactInceptionResNet(in_channels=k_soft, use_bn=use_bn, input_hw=(j_soft, j_soft), no_head=True)
        self.level_head = nn.Linear(self.soft_backbone.out_features, 2)
        self.base_scale_pA = base_scale_pA
        self.amp_min_pA = amp_min_pA
        feat = self.hard_backbone.out_features + self.soft_backbone.out_features   # no +2 -- this is v1
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in range(num_outputs)])
        else:
            self.head = nn.Linear(feat, num_outputs)

    def soft_stage(self, x_raw: torch.Tensor):
        logH = self.softdwell(x_raw)
        f_soft = self.soft_backbone(logH)
        raw = self.level_head(f_soft)
        baseline_hat_pA = torch.tanh(raw[:, 0]) * self.base_scale_pA
        amp_hat_pA = self.amp_min_pA + F.softplus(raw[:, 1])
        return f_soft, baseline_hat_pA, amp_hat_pA

    def finish(self, f_soft: torch.Tensor, hard_hist: torch.Tensor,
               baseline_hat_pA: torch.Tensor, amp_hat_pA: torch.Tensor) -> torch.Tensor:
        """baseline_hat_pA/amp_hat_pA accepted for call-site parity with v2 but UNUSED -- this is v1."""
        f_hard = self.hard_backbone(hard_hist)
        f = torch.cat([f_hard, f_soft], dim=1)
        if self.separate_heads:
            return torch.cat([h(f) for h in self.heads], dim=1)
        return self.head(f)


class HardSelfCalibratingFusionNet(nn.Module):
    """Same self-calibration idea as SelfCalibratingFusionNet, but the level estimate that drives the
    per-row threshold recentering comes from a FIXED uniform hard-threshold branch instead of SoftDwell --
    tests whether the learnable/soft part is actually necessary to get a useful level estimate, or
    whether a cheap fixed-ladder histogram alone is enough. No live O(T) computation anywhere in this
    architecture (both histograms are grind_hist_gpu calls), so it should train close to plain
    hard-threshold speed, not SoftDwell speed.

      1. fixed_stage(fixed_hist) -> f_fixed, baseline_hat_pA, amp_hat_pA   (fixed_hist precomputed ONCE,
         same uniform K/J ladder as ordinary hard-threshold training -- cheap, no per-batch rebuild)
      2. [caller builds the per-row self-calibrated hard histogram from baseline_hat_pA.detach(),
         amp_hat_pA.detach(), via grind_hist_gpu -- same per-row-ladder mechanism as
         SelfCalibratingFusionNet, just fed a different source for the level estimate]
      3. finish(f_fixed, selfcal_hist, baseline_hat_pA, amp_hat_pA) -> 9 outputs (non-detached level,
         explicitly fed to the head -- same fix as SelfCalibratingFusionNet v2)
    """
    def __init__(self, k_fixed: int, j_fixed: int, k_selfcal: int, j_selfcal: int,
                 num_outputs: int = 9, use_bn: bool = False, separate_heads: bool = False,
                 base_scale_pA: float = 2.0, amp_min_pA: float = 1.0, backbone: str = "inception",
                 convnext_hidden: int = 64, convnext_depth: int = 4, convnext_kernel_size: int = 7):
        super().__init__()
        assert backbone in ("inception", "small_cnn", "small_cnn_3d", "convnext_v2", "convnext_v2_nogrn"), backbone
        if backbone in ("convnext_v2", "convnext_v2_nogrn"):
            # convnext_hidden/depth/kernel_size: hyperparameter search (doc 11) over the champion's
            # backbone capacity/receptive-field knobs, shared by both branches (only K/J differs between
            # them). use_grn=False for the nogrn variant -- isolates GRN's own contribution (doc 10 ablation)
            # from the depthwise-conv + inverted-bottleneck-MLP design it's layered on top of.
            Backbone = functools.partial(ConvNeXtV2Backbone, hidden=convnext_hidden, depth=convnext_depth,
                                          kernel_size=convnext_kernel_size, use_grn=(backbone != "convnext_v2_nogrn"))
        else:
            Backbone = {"inception": TFExactInceptionResNet, "small_cnn": SmallCNNBackbone,
                        "small_cnn_3d": SmallCNN3DBackbone}[backbone]
        self.fixed_backbone = Backbone(in_channels=k_fixed, use_bn=use_bn, input_hw=(j_fixed, j_fixed), no_head=True)
        self.selfcal_backbone = Backbone(in_channels=k_selfcal, use_bn=use_bn, input_hw=(j_selfcal, j_selfcal), no_head=True)
        self.level_head = nn.Linear(self.fixed_backbone.out_features, 2)
        self.base_scale_pA = base_scale_pA
        self.amp_min_pA = amp_min_pA
        feat = self.fixed_backbone.out_features + self.selfcal_backbone.out_features + 2
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in range(num_outputs)])
        else:
            self.head = nn.Linear(feat, num_outputs)

    def fixed_stage(self, fixed_hist: torch.Tensor):
        """fixed_hist: (B, K_fixed, J_fixed, J_fixed) log10 counts, precomputed once, fixed uniform ladder."""
        f_fixed = self.fixed_backbone(fixed_hist)
        raw = self.level_head(f_fixed)
        baseline_hat_pA = torch.tanh(raw[:, 0]) * self.base_scale_pA
        amp_hat_pA = self.amp_min_pA + F.softplus(raw[:, 1])
        return f_fixed, baseline_hat_pA, amp_hat_pA

    def finish(self, f_fixed: torch.Tensor, selfcal_hist: torch.Tensor,
               baseline_hat_pA: torch.Tensor, amp_hat_pA: torch.Tensor) -> torch.Tensor:
        f_selfcal = self.selfcal_backbone(selfcal_hist)
        level = torch.stack([baseline_hat_pA, amp_hat_pA], dim=1)
        f = torch.cat([f_fixed, f_selfcal, level], dim=1)
        if self.separate_heads:
            return torch.cat([h(f) for h in self.heads], dim=1)
        return self.head(f)


class SoftSelfCalibratingFusionNet(nn.Module):
    """Level estimate comes from a FIXED hard-threshold branch (cheap, precomputed once); that estimate
    sets PER-ROW theta positions fed directly into the still-learnable SoftDwell layer via its existing
    per-sample theta override (softdwell.py's ChannelWiseSoftDwell/SoftDwell2DLayer already support
    theta: (B,K) -- this predates this class, likely built for an earlier freeze/refresh-style experiment,
    and is reused as-is here). tau stays the single GLOBAL learned parameter -- only detector POSITION is
    recalibrated per row, not softness width (a per-row tau is a natural extension, not built here yet).

    Unlike HardSelfCalibratingFusionNet, this is FULLY DIFFERENTIABLE end to end: no detach anywhere,
    because SoftDwell's per-sample-theta path is itself a differentiable graded function (not a hard
    step-function histogram), so gradient from the MAIN 9-output loss flows all the way back through
    theta_row into the level estimate and the fixed branch's backbone -- the auxiliary level loss is
    now a genuine auxiliary signal (extra help), not the ONLY signal reaching level_head like in the
    hard-hard variant. Single-stage forward (no two-step split needed) for the same reason.
    """
    def __init__(self, softdwell_layer: nn.Module, k_fixed: int, j_fixed: int, k_soft: int, j_soft: int,
                 num_outputs: int = 9, use_bn: bool = False, separate_heads: bool = False,
                 base_scale_pA: float = 2.0, amp_min_pA: float = 1.0, margin_pA: float = 1.0,
                 norm_min: float = -546.13, norm_max: float = 7099.7, counts_per_pA: float = 327.68):
        super().__init__()
        self.softdwell = softdwell_layer
        self.fixed_backbone = TFExactInceptionResNet(in_channels=k_fixed, use_bn=use_bn, input_hw=(j_fixed, j_fixed), no_head=True)
        self.soft_backbone = TFExactInceptionResNet(in_channels=k_soft, use_bn=use_bn, input_hw=(j_soft, j_soft), no_head=True)
        self.level_head = nn.Linear(self.fixed_backbone.out_features, 2)
        self.base_scale_pA, self.amp_min_pA, self.margin_pA = base_scale_pA, amp_min_pA, margin_pA
        self.norm_min, self.norm_max, self.counts_per_pA = norm_min, norm_max, counts_per_pA
        self.k_soft = k_soft
        feat = self.fixed_backbone.out_features + self.soft_backbone.out_features + 2
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in range(num_outputs)])
        else:
            self.head = nn.Linear(feat, num_outputs)

    def forward(self, fixed_hist: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """fixed_hist: (B, K_fixed, J_fixed, J_fixed) log10 counts, precomputed once, fixed uniform ladder.
        x_raw: (B, 1, T) RAW COUNTS (not pA -- ChannelWiseSoftDwell's norm_min/norm_max are in counts)."""
        f_fixed = self.fixed_backbone(fixed_hist)
        raw = self.level_head(f_fixed)
        baseline_hat_pA = torch.tanh(raw[:, 0]) * self.base_scale_pA
        amp_hat_pA = self.amp_min_pA + F.softplus(raw[:, 1])

        # per-row theta: K_soft positions spanning [baseline_hat - margin, baseline_hat + amp_hat + margin]
        # in pA, converted to softdwell's own normalized [0,1]-ish space (same window convention the
        # layer's global raw_theta already lives in) -- NOT detached, unlike the hard-hard variant.
        lo_counts = (baseline_hat_pA - self.margin_pA) * self.counts_per_pA
        hi_counts = (baseline_hat_pA + amp_hat_pA + self.margin_pA) * self.counts_per_pA
        t = torch.linspace(0.0, 1.0, self.k_soft, device=fixed_hist.device).unsqueeze(0)         # (1,K)
        thr_counts_row = lo_counts.unsqueeze(1) + (hi_counts - lo_counts).unsqueeze(1) * t         # (B,K)
        theta_row = (thr_counts_row - self.norm_min) / (self.norm_max - self.norm_min)             # (B,K)

        logH = self.softdwell(x_raw, theta=theta_row)   # tau=None -> uses the global learned raw_tau
        f_soft = self.soft_backbone(logH)

        level = torch.stack([baseline_hat_pA, amp_hat_pA], dim=1)
        f = torch.cat([f_fixed, f_soft, level], dim=1)
        if self.separate_heads:
            return torch.cat([h(f) for h in self.heads], dim=1)
        return self.head(f)


class FullSelfCalibratingFusionNet(nn.Module):
    """Like SoftSelfCalibratingFusionNet, but the fixed hard branch also estimates NOISE (sigma), and that
    estimate sets a PER-ROW tau (detector softness/edge width) in addition to the per-row theta (detector
    positions) SoftSelfCalibratingFusionNet already computes -- both of SoftDwell's per-sample overrides
    (theta: (B,K), tau: (B,K)) are now driven by the level estimate, not just position. Motivation: the
    SNR-binned LOO analysis (08_leave_one_out.md) found SoftDwell's sigma-estimation disadvantage WIDENS at
    high SNR (small true sigma), consistent with a fixed global tau acting as a floor that dominates once
    true sigma shrinks below it -- a per-row tau tied to the row's own estimated sigma should let the
    detector's softness track the row's actual noise instead of one dataset-wide compromise value.

    theta/tau both live in SoftDwell's own normalized [0,1]-ish space (x_norm = (x - norm_min) / (norm_max
    - norm_min)); sigma_hat_pA needs the same pA -> counts -> normalized conversion as the theta window:
        sigma_norm_hat = sigma_hat_pA * counts_per_pA / (norm_max - norm_min)
        tau_row        = tau_floor + softplus(tau_scale_raw) * sigma_norm_hat
    tau_scale_raw is a single learnable scalar (not per-detector -- keeps this a direct extension of the
    existing global tau_init convention rather than a new per-detector free parameter); tau_floor keeps
    tau_row bounded away from 0 for numerical stability (u = (x_norm - theta) / tau blows up as tau -> 0).

    Fully differentiable end to end like SoftSelfCalibratingFusionNet, same single-stage forward.
    """
    def __init__(self, softdwell_layer: nn.Module, k_fixed: int, j_fixed: int, k_soft: int, j_soft: int,
                 num_outputs: int = 9, use_bn: bool = False, separate_heads: bool = False,
                 base_scale_pA: float = 2.0, amp_min_pA: float = 1.0, sigma_min_pA: float = 0.05,
                 margin_pA: float = 1.0, tau_floor: float = 1e-3, tau_scale_init: float = 0.35,
                 norm_min: float = -546.13, norm_max: float = 7099.7, counts_per_pA: float = 327.68):
        super().__init__()
        self.softdwell = softdwell_layer
        self.fixed_backbone = TFExactInceptionResNet(in_channels=k_fixed, use_bn=use_bn, input_hw=(j_fixed, j_fixed), no_head=True)
        self.soft_backbone = TFExactInceptionResNet(in_channels=k_soft, use_bn=use_bn, input_hw=(j_soft, j_soft), no_head=True)
        self.level_head = nn.Linear(self.fixed_backbone.out_features, 3)   # baseline, amp, sigma
        self.base_scale_pA, self.amp_min_pA, self.sigma_min_pA = base_scale_pA, amp_min_pA, sigma_min_pA
        self.margin_pA, self.tau_floor = margin_pA, tau_floor
        self.norm_min, self.norm_max, self.counts_per_pA = norm_min, norm_max, counts_per_pA
        self.k_soft = k_soft
        # softplus(tau_scale_raw) starts at ~tau_scale_init: inverse-softplus(y) = log(exp(y)-1)
        init_raw = math.log(math.expm1(max(tau_scale_init, 1e-6)))
        self.tau_scale_raw = nn.Parameter(torch.tensor(init_raw, dtype=torch.float32))
        feat = self.fixed_backbone.out_features + self.soft_backbone.out_features + 3   # +3: baseline, amp, sigma
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in range(num_outputs)])
        else:
            self.head = nn.Linear(feat, num_outputs)

    def forward(self, fixed_hist: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """fixed_hist: (B, K_fixed, J_fixed, J_fixed) log10 counts, precomputed once, fixed uniform ladder.
        x_raw: (B, 1, T) RAW COUNTS (not pA -- ChannelWiseSoftDwell's norm_min/norm_max are in counts)."""
        f_fixed = self.fixed_backbone(fixed_hist)
        raw = self.level_head(f_fixed)
        baseline_hat_pA = torch.tanh(raw[:, 0]) * self.base_scale_pA
        amp_hat_pA = self.amp_min_pA + F.softplus(raw[:, 1])
        sigma_hat_pA = self.sigma_min_pA + F.softplus(raw[:, 2])

        lo_counts = (baseline_hat_pA - self.margin_pA) * self.counts_per_pA
        hi_counts = (baseline_hat_pA + amp_hat_pA + self.margin_pA) * self.counts_per_pA
        t = torch.linspace(0.0, 1.0, self.k_soft, device=fixed_hist.device).unsqueeze(0)         # (1,K)
        thr_counts_row = lo_counts.unsqueeze(1) + (hi_counts - lo_counts).unsqueeze(1) * t         # (B,K)
        theta_row = (thr_counts_row - self.norm_min) / (self.norm_max - self.norm_min)             # (B,K)

        sigma_norm_hat = sigma_hat_pA * self.counts_per_pA / (self.norm_max - self.norm_min)        # (B,)
        tau_row = self.tau_floor + F.softplus(self.tau_scale_raw) * sigma_norm_hat                  # (B,)
        tau_row = tau_row.unsqueeze(1).expand(-1, self.k_soft)                                       # (B,K), same tau across detectors

        logH = self.softdwell(x_raw, theta=theta_row, tau=tau_row)
        f_soft = self.soft_backbone(logH)

        level = torch.stack([baseline_hat_pA, amp_hat_pA, sigma_hat_pA], dim=1)
        f = torch.cat([f_fixed, f_soft, level], dim=1)
        if self.separate_heads:
            out = torch.cat([h(f) for h in self.heads], dim=1)
        else:
            out = self.head(f)
        return out, baseline_hat_pA, amp_hat_pA, sigma_hat_pA


class LearnedSelfCalibratingFusionNet(nn.Module):
    """Same hard-fixed-branch-gauges-SoftDwell idea as FullSelfCalibratingFusionNet, but theta/tau are no
    longer a hand-derived FORMULA from 3 scalar estimates (baseline, amp, sigma) -- they come from real
    LEARNED heads (theta_head, tau_head: plain nn.Linear) that consume the fixed branch's full feature
    vector (fixed_backbone's conv features, the same richness the final head itself sees), not just 3
    numbers. Same offset convention as the pre-existing AdaptiveSoftDwellNet (adaptive.py), which predates
    this class and established the pattern -- additive offset on theta, multiplicative (exp) offset on
    tau, both heads zero-initialized so the model starts at the plain global theta/tau and only departs
    from it as training finds it useful:
        theta_row = raw_theta + theta_head(f_fixed)            (B,K), zero-init -> starts at raw_theta
        tau_row   = raw_tau * exp(tau_head(f_fixed))            (B,K), zero-init -> starts at raw_tau
    Unlike AdaptiveSoftDwellNet, f_fixed comes from a proper per-row-self-calibratable dwell-histogram
    branch (TFExactInceptionResNet on a fixed uniform ladder), not the cheap raw-value amplitude
    histogram -- richer input, and consistent with every other class in this family.

    level_head (baseline/amp/sigma) is kept ONLY as an auxiliary output for the small aux loss and as an
    explicit signal concatenated into the final head (same as every other variant) -- it no longer
    participates in computing theta/tau at all, that's now purely the learned heads' job.
    """
    def __init__(self, softdwell_layer: nn.Module, k_fixed: int, j_fixed: int, k_soft: int, j_soft: int,
                 num_outputs: int = 9, use_bn: bool = False, separate_heads: bool = False,
                 base_scale_pA: float = 2.0, amp_min_pA: float = 1.0, sigma_min_pA: float = 0.05):
        super().__init__()
        self.softdwell = softdwell_layer
        self.fixed_backbone = TFExactInceptionResNet(in_channels=k_fixed, use_bn=use_bn, input_hw=(j_fixed, j_fixed), no_head=True)
        self.soft_backbone = TFExactInceptionResNet(in_channels=k_soft, use_bn=use_bn, input_hw=(j_soft, j_soft), no_head=True)
        self.level_head = nn.Linear(self.fixed_backbone.out_features, 3)   # baseline, amp, sigma -- aux output only
        self.theta_head = nn.Linear(self.fixed_backbone.out_features, k_soft)
        self.tau_head = nn.Linear(self.fixed_backbone.out_features, k_soft)
        for m in (self.theta_head, self.tau_head):
            nn.init.zeros_(m.weight); nn.init.zeros_(m.bias)
        self.base_scale_pA, self.amp_min_pA, self.sigma_min_pA = base_scale_pA, amp_min_pA, sigma_min_pA
        feat = self.fixed_backbone.out_features + self.soft_backbone.out_features + 3
        self.separate_heads = separate_heads
        if separate_heads:
            self.heads = nn.ModuleList([nn.Linear(feat, 1) for _ in range(num_outputs)])
        else:
            self.head = nn.Linear(feat, num_outputs)

    def forward(self, fixed_hist: torch.Tensor, x_raw: torch.Tensor) -> torch.Tensor:
        """fixed_hist: (B, K_fixed, J_fixed, J_fixed) log10 counts, precomputed once, fixed uniform ladder.
        x_raw: (B, 1, T) RAW COUNTS (not pA -- ChannelWiseSoftDwell's norm_min/norm_max are in counts)."""
        f_fixed = self.fixed_backbone(fixed_hist)
        raw = self.level_head(f_fixed)
        baseline_hat_pA = torch.tanh(raw[:, 0]) * self.base_scale_pA
        amp_hat_pA = self.amp_min_pA + F.softplus(raw[:, 1])
        sigma_hat_pA = self.sigma_min_pA + F.softplus(raw[:, 2])

        inner = self.softdwell.inner
        theta_row = inner.raw_theta.unsqueeze(0) + self.theta_head(f_fixed)             # (B,K)
        tau_row = inner.raw_tau.unsqueeze(0) * torch.exp(self.tau_head(f_fixed))        # (B,K)

        logH = self.softdwell(x_raw, theta=theta_row, tau=tau_row)
        f_soft = self.soft_backbone(logH)

        level = torch.stack([baseline_hat_pA, amp_hat_pA, sigma_hat_pA], dim=1)
        f = torch.cat([f_fixed, f_soft, level], dim=1)
        if self.separate_heads:
            out = torch.cat([h(f) for h in self.heads], dim=1)
        else:
            out = self.head(f)
        return out, baseline_hat_pA, amp_hat_pA, sigma_hat_pA
