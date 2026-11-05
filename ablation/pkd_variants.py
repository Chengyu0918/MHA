"""Structural ablation variants of the production PKDFusion backbone (01_MHA).

Subclasses only -- production code is never modified. The launcher swaps the
pkd.fusion.PKDFusion symbol before pkd.runtime.train() imports it, so training
protocol, data pipeline, EMA and provenance machinery are identical to h1/h4.

Arms (all on the h4 config, num_heads=4, allow_multihead=true):
  noattn : DFU keeps its gating structure but every global attention() call is
           replaced by identity (no cross-token mixing). Same modules and same
           initialization as h4; MPAM banks receive no gradient (dead params).
  nodfu  : the whole DFU module is bypassed; decode3 consumes psi directly
           (z is computed but unused).
  coarse : DFU retained; both fine decoder stages (skip connections to s2/s1)
           removed, new 1x1 head maps width channels to fused Y at full res.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CODE = '/home/stu5/chengyu0918/LESAFusion_TAES_20261028/code'
if CODE not in sys.path:
    sys.path.insert(0, CODE)
from pkd.fusion import PKDFusion


def _dfu_noattn_forward(dfu, psi, z):
    """DFU.forward with attention()==identity; reuses dfu's own gate convs."""
    q, k, v = psi, psi, psi                      # MPAM bypassed, banks unused
    joint = torch.cat((q, k, v), dim=1)
    gf = torch.sigmoid(dfu.forget_gate(joint).float())
    gi = torch.sigmoid(dfu.input_gate(joint).float()).clamp_min(1e-8)
    d_raw = gf + gi.log()
    d_max = d_raw.amax(dim=1, keepdim=True)
    d = (d_raw - d_max).exp()
    c = d                                        # attention(d,q,k) -> identity
    eta = torch.maximum(c.sum(dim=1, keepdim=True).abs(), (-d_max).exp())
    h = (c / eta.clamp_min(1e-12)) * v.float()
    return (torch.sigmoid(dfu.output_gate(z).float()) * h).to(psi.dtype)


class NoAttnFusion(PKDFusion):
    """w/o global attention: identical init to h4, forward path ablated."""
    def forward_with_aux(self, vi_y, ir_y):
        if vi_y.shape != ir_y.shape or vi_y.ndim != 4 or vi_y.shape[1] != 1:
            raise ValueError('vi_y and ir_y must have identical B,1,H,W shapes')
        s1, s2, s3 = self._encode(torch.cat((ir_y, vi_y), dim=1))
        psi, z = self.joint_projection(s3).chunk(2, dim=1)
        psi = self.shallow(psi)
        decoded = self.decode3(_dfu_noattn_forward(self.dfu, psi, z))
        decoded = F.interpolate(decoded, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        decoded = self.decode2(torch.cat((decoded, s2), dim=1))
        decoded = F.interpolate(decoded, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        decoded = self.decode1(torch.cat((decoded, s1), dim=1))
        base = torch.sigmoid(self.out(decoded))
        zero = torch.zeros_like(base)
        return base, {'base': base, 'local_delta': zero, 'local_gate': zero,
                      'support': zero, 'residual': zero, 'delta': zero, 'gate': zero}


class NoDFUFusion(PKDFusion):
    """w/o DFU: psi feeds decode3 directly (z unused)."""
    def forward_with_aux(self, vi_y, ir_y):
        if vi_y.shape != ir_y.shape or vi_y.ndim != 4 or vi_y.shape[1] != 1:
            raise ValueError('vi_y and ir_y must have identical B,1,H,W shapes')
        s1, s2, s3 = self._encode(torch.cat((ir_y, vi_y), dim=1))
        psi, z = self.joint_projection(s3).chunk(2, dim=1)
        psi = self.shallow(psi)
        decoded = self.decode3(psi)              # DFU bypassed
        decoded = F.interpolate(decoded, size=s2.shape[-2:], mode='bilinear', align_corners=False)
        decoded = self.decode2(torch.cat((decoded, s2), dim=1))
        decoded = F.interpolate(decoded, size=s1.shape[-2:], mode='bilinear', align_corners=False)
        decoded = self.decode1(torch.cat((decoded, s1), dim=1))
        base = torch.sigmoid(self.out(decoded))
        zero = torch.zeros_like(base)
        return base, {'base': base, 'local_delta': zero, 'local_gate': zero,
                      'support': zero, 'residual': zero, 'delta': zero, 'gate': zero}


class CoarseOnlyFusion(PKDFusion):
    """coarse-only: DFU retained, fine decoder stages/skips removed."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.out_coarse = nn.Conv2d(self.config['width'], 1, 1)

    def forward_with_aux(self, vi_y, ir_y):
        if vi_y.shape != ir_y.shape or vi_y.ndim != 4 or vi_y.shape[1] != 1:
            raise ValueError('vi_y and ir_y must have identical B,1,H,W shapes')
        s1, s2, s3 = self._encode(torch.cat((ir_y, vi_y), dim=1))
        psi, z = self.joint_projection(s3).chunk(2, dim=1)
        psi = self.shallow(psi)
        decoded = self.decode3(self.dfu(psi, z))
        decoded = F.interpolate(decoded, size=vi_y.shape[-2:], mode='bilinear', align_corners=False)
        base = torch.sigmoid(self.out_coarse(decoded))
        zero = torch.zeros_like(base)
        return base, {'base': base, 'local_delta': zero, 'local_gate': zero,
                      'support': zero, 'residual': zero, 'delta': zero, 'gate': zero}


VARIANTS = {'noattn': NoAttnFusion, 'nodfu': NoDFUFusion, 'coarse': CoarseOnlyFusion}
