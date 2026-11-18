# scripts/refine_pipeline/model.py
"""
SimRefineNet: a RAFT-style recurrent update operator that refines the estimate of an already trained network
by comparing the recorded trace with a re-simulation at the current estimate.

    theta_0 = stage-1 network(trace)                               (frozen, e.g. HardSelfCalibratingFusionNet)
    h_0, inp = context(stage-1 features)
    for k = 0 .. N-1:
        trace_k = simulator(theta_k)                               (CPU, not differentiable)
        m_k     = encoder([H_obs, H_sim_k, H_obs - H_sim_k])       ("correlation" of RAFT: observed vs re-simulated)
        h_{k+1} = GRU([m_k, embed(theta_k), inp], h_k)
        theta_{k+1} = theta_k + delta(h_{k+1})

As in RAFT, the weights are shared over the iterations, theta_k enters each step detached (here it has to be: the
simulator has no gradient), the hidden state carries the gradient through the unrolled loop, and the loss is a
weighted sum over all iterates with more weight on the later ones. The last layer of delta() starts at zero, so an
untrained refiner returns the stage-1 estimate unchanged.

theta is the stage-1 output space: log10 rates followed by the standardized conditions (baseline, amplitude,
sigma), the same space the stage-1 loss is computed in.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.scripts.softdwell_pipeline.model import ConvNeXtV2Backbone


class SimRefineNet(nn.Module):
    def __init__(self, hist_channels: int, hist_hw, theta_dim: int, ctx_dim: int, theta_center=None,
                 theta_scale=None, hidden: int = 128, enc_hidden: int = 64, enc_depth: int = 3,
                 kernel_size: int = 7, theta_embed: int = 64, max_step: float = 1.0):
        """
        hist_channels: channels of ONE histogram stack (observed or simulated); the encoder sees three of them.
        theta_center / theta_scale (theta_dim,): theta is fed to the embedding as (theta - center) / scale.
        max_step: largest change of one theta component per iteration (soft limit, tanh).
        """
        super().__init__()
        self.encoder = ConvNeXtV2Backbone(in_channels=3 * hist_channels, input_hw=tuple(hist_hw), no_head=True,
                                          hidden=enc_hidden, depth=enc_depth, kernel_size=kernel_size)
        self.theta_enc = nn.Sequential(nn.Linear(theta_dim, theta_embed), nn.GELU(),
                                       nn.Linear(theta_embed, theta_embed))
        self.ctx_proj = nn.Sequential(nn.LayerNorm(ctx_dim), nn.Linear(ctx_dim, 2 * hidden))
        self.gru = nn.GRUCell(self.encoder.out_features + theta_embed + hidden, hidden)
        self.delta = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, theta_dim))
        nn.init.zeros_(self.delta[-1].weight)
        nn.init.zeros_(self.delta[-1].bias)
        self.max_step = float(max_step)
        c = torch.zeros(theta_dim) if theta_center is None else torch.as_tensor(theta_center, dtype=torch.float32)
        s = torch.ones(theta_dim) if theta_scale is None else torch.as_tensor(theta_scale, dtype=torch.float32)
        self.register_buffer("theta_center", c)
        self.register_buffer("theta_scale", s)

    def init_state(self, ctx: torch.Tensor):
        """ctx (B, ctx_dim) stage-1 features -> (hidden state h_0, context input), as RAFT's context encoder."""
        h, inp = self.ctx_proj(ctx).chunk(2, dim=1)
        return torch.tanh(h), F.relu(inp)

    def step(self, h, inp, theta, h_obs, h_sim):
        """One update. theta (B, theta_dim) should come in detached. Returns (h, new theta)."""
        m = self.encoder(torch.cat([h_obs, h_sim, h_obs - h_sim], dim=1))
        e = self.theta_enc((theta - self.theta_center) / self.theta_scale)
        h = self.gru(torch.cat([m, e, inp], dim=1), h)
        d = self.delta(h)
        d = self.max_step * torch.tanh(d / self.max_step)
        return h, theta + d
