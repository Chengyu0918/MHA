"""Generate annotated detection showcase tiles (real predictions, real images).

Picks test frames where h4 detects a small UAV that simple_avg and h1 both
miss (from .per_frame.json), runs each method's detector on its own fused
image at its frozen threshold, and draws GT + predictions with a zoom inset
around the small target. Output: /tmp/showcase/<sid>_<method>.jpg
"""
import json
import sys
from pathlib import Path

import torch

B = Path('/home/stu5/chengyu0918/LESAFusion_TAES_20261028/runs_mha_followup')
OUT = Path('/tmp/showcase')
OUT.mkdir(exist_ok=True)

_ = torch.zeros(1, device='cuda')  # pin CUDA context before ultralytics (CVD fix)

from ultralytics import YOLO  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

RUNS = {'avg': 'simple_avg_d0', 'h1': 'h1_f3408_d0', 'h4': 'h4_f3408_d0'}
ORDER = ['avg', 'h1', 'h4']
COLORS = {'gt': (34, 197, 94), 'avg': (120, 120, 130), 'h1': (245, 158, 11), 'h4': (37, 99, 235)}

thr = {k: json.load(open(B / 'eval' / f'{v}.json'))['frozen_threshold'] for k, v in RUNS.items()}
pf = {k: {r['sample_id']: r for r in json.load(open(B / 'eval' / f'{v}.per_frame.json'))}
      for k, v in RUNS.items()}

cands = [sid for sid, r in pf['h4'].items()
         if r['small_tp'] >= 1 and r['small_gt_boxes'] >= 1
         and pf['avg'].get(sid, {}).get('small_tp', 0) == 0
         and pf['h1'].get(sid, {}).get('small_tp', 0) == 0]
print('candidate frames:', len(cands), cands[:10])

sel, seen = [], set()
for sid in sorted(cands):
    s = sid[:2]
    if s in seen:
        continue
    seen.add(s)
    sel.append(sid)
    if len(sel) == 3:
        break
print('selected:', sel)
if not sel:
    sys.exit('no showcase frames found')

import yaml  # noqa: E402

tdirs = {}
for k, v in RUNS.items():
    a = yaml.safe_load(open(B / 'detection' / v / 'args.yaml'))
    data_p = Path(a['data']).parent
    for cand in (data_p / 'images' / 'test',
                 data_p.parent / 'export_test' / 'images' / 'test'):
        if (cand / f'{sel[0]}.png').exists():
            tdirs[k] = cand.resolve()
            break
    else:
        raise AssertionError(f'no test image dir for {k} under {data_p}')


def gt_boxes(sid):
    for k in ('h4', 'h1'):
        p = tdirs[k].parent.parent / 'labels' / 'test' / f'{sid}.txt'
        if p.exists():
            break
    W, H = Image.open(tdirs['h4'] / f'{sid}.png').size
    boxes = []
    for line in p.read_text().split('\n'):
        line = line.split()
        if len(line) < 5:
            continue
        _, cx, cy, w, h = line[:5]
        cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
        boxes.append((cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    return boxes, (W, H)


def font(sz):
    for cand in ('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
                 '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'):
        if Path(cand).exists():
            return ImageFont.truetype(cand, sz)
    return ImageFont.load_default()


def annotate(img, boxes, confs, gts, method):
    d = ImageDraw.Draw(img)
    f_small, f_tiny = font(15), font(13)
    for (x1, y1, x2, y2) in gts:
        d.rectangle([x1, y1, x2, y2], outline=COLORS['gt'], width=2)
        d.text((x1 + 2, max(0, y1 - 17)), 'GT', fill=COLORS['gt'], font=f_small)
    c = COLORS[method]
    for (x1, y1, x2, y2), cf in zip(boxes, confs):
        d.rectangle([x1, y1, x2, y2], outline=c, width=2)
        tag = f'{cf:.2f}'
        tb = d.textbbox((x1 + 2, y2 + 2), tag, font=f_tiny)
        d.rectangle([tb[0] - 1, tb[1] - 1, tb[2] + 1, tb[3] + 1], fill=c)
        d.text((x1 + 2, y2 + 2), tag, fill=(255, 255, 255), font=f_tiny)
    return img


models = {k: YOLO(str(B / 'detection' / v / 'weights' / 'best.pt')) for k, v in RUNS.items()}
meta = {}
for sid in sel:
    gts, (W, H) = gt_boxes(sid)
    meta[sid] = {'gt': gts, 'W': W, 'H': H, 'res': {}}
    for k in ORDER:
        img = Image.open(tdirs[k] / f'{sid}.png').convert('RGB')
        r = models[k].predict(str(tdirs[k] / f'{sid}.png'), conf=thr[k] - 1e-9,
                              imgsz=640, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().tolist()
        confs = r.boxes.conf.cpu().tolist()
        meta[sid]['res'][k] = {'n': len(boxes),
                               'confs': [round(c, 2) for c in confs[:5]]}
        annotate(img, boxes, confs, gts, k)
        # zoom inset around first GT box (small target), from annotated image
        if gts:
            gx1, gy1, gx2, gy2 = gts[0]
            cx, cy = (gx1 + gx2) / 2, (gy1 + gy2) / 2
            half = max(70, (max(gx2 - gx1, gy2 - gy1)) * 3 + 30)
            zx1, zy1 = max(0, cx - half), max(0, cy - half)
            zx2, zy2 = min(W, cx + half), min(H, cy + half)
            z = img.crop((int(zx1), int(zy1), int(zx2), int(zy2)))
            zf = 240 / max(1, z.width)
            z = z.resize((int(z.width * zf), int(z.height * zf)), Image.LANCZOS)
            px, py = img.width - z.width - 8, img.height - z.height - 8
            d = ImageDraw.Draw(img)
            d.rectangle([px - 2, py - 2, px + z.width + 2, py + z.height + 2],
                        outline=COLORS[k], width=3)
            img.paste(z, (px, py))
        img.save(OUT / f'{sid}_{k}.jpg', quality=88)
        print('saved', sid, k, 'preds=', len(boxes))

json.dump(meta, open(OUT / 'meta.json', 'w'), indent=1)
print('SHOWCASE_DONE')
