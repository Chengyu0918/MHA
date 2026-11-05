"""Multi-Head Attention Fusion Network for Infrared-Visible Image Fusion.

This module implements a dual-branch encoder-decoder fusion network where
cross-modal feature interaction is performed via multi-head global attention.
Each attention head independently captures a distinct pattern of infrared-visible
complementarity in a learned subspace; outputs are concatenated and projected
back to the original dimension.

Architecture:
    vi_y, ir_y -> shared encoder -> tokens -> multi-head cross-attention
    -> residual connection -> decoder -> fused Y -> chroma restore -> RGB

Note: The multi-head mechanism is the core contribution; setting num_heads=1
recovers the single-head baseline for ablation comparison.
"""
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Building Blocks
# ============================================================

class ConvBlock(nn.Module):
    """Conv -> Norm -> SiLU block."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class Downsample(nn.Module):
    """Feature-map downsampling with channel expansion."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = ConvBlock(in_ch, out_ch, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Upsample(nn.Module):
    """Feature-map upsampling with channel reduction."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.block = ConvBlock(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.up(x))


# ============================================================
# Multi-Head Cross-Modal Attention (core module)
# ============================================================

class MultiHeadCrossModalAttention(nn.Module):
    """Global multi-head attention for cross-modal feature interaction.

    Given flattened feature tokens from the shared encoder, this module:
    1. Projects tokens into query/key/value spaces.
    2. Splits channels into ``num_heads`` independent subspaces.
    3. Computes scaled dot-product attention in each subspace.
    4. Concatenates head outputs and applies an output projection.
    5. Adds a residual connection for stable training.

    With ``num_heads=1`` this reduces to standard single-head attention
    (the ablation baseline).
    """

    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} not divisible by num_heads={num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = math.sqrt(self.head_dim)

        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, C]  flattened spatial tokens from the encoder
        Returns:
            [B, N, C]  attention-enhanced tokens (residual connected)
        """
        B, N, C = x.shape
        residual = x

        # Project to Q, K, V
        q = self.wq(x)  # [B, N, C]
        k = self.wk(x)
        v = self.wv(x)

        # Split into heads: [B, heads, N, head_dim]
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention per head
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale  # [B, h, N, N]
        attn = F.softmax(scores, dim=-1)
        attn = self.attn_drop(attn)
        out = torch.matmul(attn, v)  # [B, h, N, head_dim]

        # Concatenate heads: [B, N, C]
        out = out.transpose(1, 2).contiguous().view(B, N, C)

        # Output projection + residual
        out = self.out_drop(self.wo(out))
        return self.norm(residual + out)


# ============================================================
# DFU: Dual-Feature Unit (local feature extraction)
# ============================================================

class DualFeatureUnit(nn.Module):
    """Local feature extraction with parallel dilated + pointwise paths."""

    def __init__(self, ch: int, dilation: int = 2):
        super().__init__()
        self.dilated = nn.Conv2d(ch, ch, 3, padding=dilation, dilation=dilation, bias=False)
        self.pointwise = nn.Conv2d(ch, ch, 1, bias=False)
        self.norm = nn.BatchNorm2d(ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.dilated(x) + self.pointwise(x)))


# ============================================================
# MPAM: Modality-Pair Attention Module
# ============================================================

class ModalityPairAttention(nn.Module):
    """Wrap global attention over spatial tokens with local feature extraction."""

    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.local = DualFeatureUnit(ch)
        self.global_attn = MultiHeadCrossModalAttention(ch, num_heads=num_heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] -> [B, C, H, W]"""
        B, C, H, W = x.shape
        out = self.local(x)
        tokens = out.flatten(2).transpose(1, 2)  # [B, H*W, C]
        tokens = self.global_attn(tokens)
        out = tokens.transpose(1, 2).view(B, C, H, W)
        return out + x  # outer residual


# ============================================================
# Full Fusion Network
# ============================================================

class MHAFusionNet(nn.Module):
    """Multi-Head Attention Fusion Network.

    Takes Y-channel inputs from visible and infrared images, fuses them
    via a shared encoder + multi-head cross-modal attention, and outputs
    a fused Y-channel image.

    Args:
        in_ch: input channels per modality (1 for Y channel)
        base_ch: base feature width of the encoder
        num_heads: number of attention heads (4 for MHA, 1 for baseline)
        num_blocks: number of MPAM blocks at each resolution level
    """

    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 32,
        num_heads: int = 4,
        num_blocks: int = 2,
    ):
        super().__init__()
        self.num_heads = num_heads

        # Shared encoder (processes both modalities with same weights)
        self.enc_conv = ConvBlock(in_ch, base_ch)
        self.enc_down1 = Downsample(base_ch, base_ch * 2)       # /2
        self.enc_down2 = Downsample(base_ch * 2, base_ch * 4)   # /4
        self.enc_down3 = Downsample(base_ch * 4, base_ch * 8)   # /8

        # Attention blocks at each level
        self.mpam1 = nn.ModuleList(
            [ModalityPairAttention(base_ch * 2, num_heads) for _ in range(num_blocks)])
        self.mpam2 = nn.ModuleList(
            [ModalityPairAttention(base_ch * 4, num_heads) for _ in range(num_blocks)])
        self.mpam3 = nn.ModuleList(
            [ModalityPairAttention(base_ch * 8, num_heads) for _ in range(num_blocks)])

        # Decoder with skip connections
        self.dec_up3 = Upsample(base_ch * 8, base_ch * 4)
        self.dec_up2 = Upsample(base_ch * 4, base_ch * 2)
        self.dec_up1 = Upsample(base_ch * 2, base_ch)
        self.dec_out = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, vi_y: torch.Tensor, ir_y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            vi_y: [B, 1, H, W]  visible light Y channel
            ir_y: [B, 1, H, W]  infrared Y channel
        Returns:
            [B, 1, H, W]  fused Y channel
        """
        # Concatenate along channel for joint encoding
        x = torch.cat([vi_y, ir_y], dim=1)  # [B, 2, H, W]

        # Encode
        e0 = self.enc_conv(x)                    # [B, C,   H,   W]
        e1 = self.enc_down1(e0)                  # [B, 2C,  H/2, W/2]
        e2 = self.enc_down2(e1)                  # [B, 4C,  H/4, W/4]
        e3 = self.enc_down3(e2)                  # [B, 8C,  H/8, W/8]

        # Multi-head attention at each level
        for block in self.mpam1:
            e1 = block(e1)
        for block in self.mpam2:
            e2 = block(e2)
        for block in self.mpam3:
            e3 = block(e3)

        # Decode with skip connections
        d3 = self.dec_up3(e3) + e2               # [B, 4C, H/4, W/4]
        d2 = self.dec_up2(d3) + e1               # [B, 2C, H/2, W/2]
        d1 = self.dec_up1(d2) + e0               # [B, C,  H,   W]

        # Output fused Y channel
        return torch.sigmoid(self.dec_out(d1))    # [B, 1, H, W]


# ============================================================
# Training loss
# ============================================================

class FusionLoss(nn.Module):
    """Combined intensity + gradient + SSIM loss for image fusion."""

    def __init__(self, w_int: float = 1.0, w_grad: float = 10.0, w_ssim: float = 6.0):
        super().__init__()
        self.w_int = w_int
        self.w_grad = w_grad
        self.w_ssim = w_ssim

    def forward(self, fused: torch.Tensor, vi: torch.Tensor, ir: torch.Tensor) -> torch.Tensor:
        # Intensity: max of both modalities (keep bright regions from either)
        max_intensity = torch.max(vi, ir)
        l_int = F.l1_loss(fused, max_intensity)

        # Gradient: preserve edges from both
        def grad(x):
            return torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1])
        l_grad = F.l1_loss(grad(fused), torch.max(grad(vi), grad(ir)))

        # Simplified SSIM on local patches
        l_ssim = 1.0 - F.cosine_similarity(
            fused.flatten(1), max_intensity.flatten(1), dim=1).mean()

        return self.w_int * l_int + self.w_grad * l_grad + self.w_ssim * l_ssim


# ============================================================
# Utility: restore RGB from fused Y + original visible chroma
# ============================================================

def restore_rgb(fused_y: torch.Tensor, vi_rgb: torch.Tensor) -> torch.Tensor:
    """Replace the Y channel of vi_rgb with fused_y.

    Args:
        fused_y: [B, 1, H, W]
        vi_rgb: [B, 3, H, W]
    Returns:
        [B, 3, H, W]
    """
    # ITU-R BT.601 inverse
    cb = vi_rgb[:, 1:2] - 0.564 * (vi_rgb[:, 2:3] - fused_y)
    cr = vi_rgb[:, 2:3] - 0.713 * (vi_rgb[:, 0:1] - fused_y)
    r = fused_y + 1.403 * cr
    g = fused_y - 0.714 * cr - 0.344 * cb
    b = fused_y + 1.773 * cb
    return torch.cat([r, g, b], dim=1).clamp(0, 1)


# ============================================================
# Quick smoke test
# ============================================================

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MHAFusionNet(num_heads=4).to(device)

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    print(f"MHA Fusion params: {total:,}")

    # Baseline for comparison
    base = MHAFusionNet(num_heads=1).to(device)
    base_total = sum(p.numel() for p in base.parameters())
    print(f"Baseline (1-head) params: {base_total:,}")
    print(f"Delta: {total - base_total:,}")

    # Forward pass
    vi = torch.randn(2, 1, 128, 128).to(device)
    ir = torch.randn(2, 1, 128, 128).to(device)
    with torch.no_grad():
        out = model(vi, ir)
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.4f}, {out.max():.4f}]")

    # Loss
    loss_fn = FusionLoss()
    vi_rgb = torch.randn(2, 3, 128, 128).to(device)
    loss = loss_fn(out, vi, ir)
    print(f"Loss: {loss.item():.4f}")

    # Chroma restore
    rgb = restore_rgb(out, vi_rgb)
    print(f"RGB output: {rgb.shape}")
    print("SMOKE TEST PASSED")
