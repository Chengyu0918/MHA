"""Inference wrappers for external fusion baselines (P3-C).

Wraps CDDFuse (CVPR'23, IVF branch) and SwinFusion (ICBRA'22, IVF branch)
pretrained checkpoints to the pkd forward interface: forward(vi_y, ir_y) ->
fused Y in [0,1], so pkd.runtime.fuse_pair handles RGB<->Y and chroma restore
exactly as for the h1/h4/variant models. Each wrapper follows the original
repo's released test-script conventions (padding, argument order, output
normalization) as closely as possible.
"""
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

CDD = '/tmp/p3c_repos/CDDFuse'
SWIN = '/tmp/p3c_repos/SwinFusion'


def _pad8(x):
    h, w = x.shape[-2:]
    ph = (h + 7) // 8 * 8 - h
    pw = (w + 7) // 8 * 8 - w
    return F.pad(x, (0, pw, 0, ph), mode='reflect'), h, w


class CDDFuseNet(nn.Module):
    """CDDFuse_IVF.pth wrapped as forward(vi_y, ir_y).

    Mirrors test_IVF.py: Encoder on both Y inputs, feature-sum fusion layers,
    Decoder(vi, F_B, F_D), per-image min-max normalization. DataParallel
    wrappers keep checkpoint key compatibility ('module.' prefix).
    """
    def __init__(self, ckpt_path, device='cuda'):
        super().__init__()
        if CDD not in sys.path:
            sys.path.insert(0, CDD)
        from net import (Restormer_Encoder, Restormer_Decoder,
                         BaseFeatureExtraction, DetailFeatureExtraction)
        self.enc = nn.DataParallel(Restormer_Encoder()).to(device)
        self.dec = nn.DataParallel(Restormer_Decoder()).to(device)
        self.base = nn.DataParallel(BaseFeatureExtraction(dim=64, num_heads=8)).to(device)
        self.det = nn.DataParallel(DetailFeatureExtraction(num_layers=1)).to(device)
        ck = torch.load(ckpt_path, map_location=device)
        # released IVF weights use legacy DIDF_* keys; current test script says CDDF_*
        self.enc.load_state_dict(ck.get('CDDF_Encoder', ck['DIDF_Encoder']))
        self.dec.load_state_dict(ck.get('CDDF_Decoder', ck['DIDF_Decoder']))
        self.base.load_state_dict(ck['BaseFuseLayer'])
        self.det.load_state_dict(ck['DetailFuseLayer'])
        for m in (self.enc, self.dec, self.base, self.det):
            m.eval()

    @torch.no_grad()
    def forward(self, vi_y, ir_y):
        vi_pad, h, w = _pad8(vi_y)
        ir_pad, _, _ = _pad8(ir_y)
        fvb, fvd, _ = self.enc(vi_pad)
        fib, fid, _ = self.enc(ir_pad)
        fb = self.base(fvb + fib)
        fd = self.det(fvd + fid)
        out, _ = self.dec(vi_pad, fb, fd)
        out = out[..., :h, :w]
        mn, mx = out.min(), out.max()
        return (out - mn) / (mx - mn).clamp_min(1e-8)


class SwinFusionNet(nn.Module):
    """SwinFusion IVF 10000_E.pth wrapped as forward(vi_y, ir_y).

    Mirrors test_swinfusion.py: net(ir, vi) argument order, reflect pad to a
    multiple of window_size=8, output cropped back (no extra normalization;
    saving clamps to [0,1] like their tensor2uint).
    """
    def __init__(self, ckpt_dir, device='cuda'):
        super().__init__()
        if SWIN not in sys.path:
            sys.path.insert(0, SWIN)
        from models.network_swinfusion1 import SwinFusion as Net
        self.net = Net(upscale=1, in_chans=1, img_size=128, window_size=8, img_range=1.,
                       depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                       mlp_ratio=2, upsampler=None, resi_connection='1conv')
        ck = torch.load(str(ckpt_dir).rstrip('/') + '/10000_E.pth', map_location='cpu')
        self.net.load_state_dict(ck['params'] if 'params' in ck else ck, strict=True)
        self.net = self.net.to(device).eval()

    @torch.no_grad()
    def forward(self, vi_y, ir_y):
        a, h, w = _pad8(ir_y)
        b, _, _ = _pad8(vi_y)
        out = self.net(a, b)
        return out[..., :h, :w]


def build_baseline(method, device='cuda'):
    if method == 'cddfuse':
        return CDDFuseNet(CDD + '/models/CDDFuse_IVF.pth', device)
    if method == 'swinfusion':
        return SwinFusionNet(SWIN + '/Model/Infrared_Visible_Fusion/Infrared_Visible_Fusion/models', device)
    raise ValueError('unknown baseline: ' + method)
