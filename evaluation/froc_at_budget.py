"""Round 3 P3: common FP-budget FROC for ablation arms + external baselines.

Re-runs full-confidence predictions from each detector checkpoint on its own
fused test images, sweeps thresholds, and reports max small-target recall at
10/25/50/100 FP per 1000 frames. Same protocol as round3_p2v2 (which covered
h1/h4/avg). CUDA context pre-init pins the GPU (select_device CVD fix).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE = Path('/home/stu5/chengyu0918/LESAFusion_TAES_20261028/code')
PKG = CODE.parent
sys.path.insert(0, str(CODE))
sys.path.insert(0, '/tmp')

_ = torch.zeros(1, device='cuda')  # pin device snapshot before ultralytics

from pkd.evaluation import _results, match_frame  # noqa: E402
from pkd.detection import load_detector  # noqa: E402
from pkd.data import load_manifest  # noqa: E402

OUT = PKG / 'runs_mha_round3'
RUNS = PKG / 'runs_mha_followup'
BUDGETS = [10, 25, 50, 100]
MANIFEST = load_manifest(str(PKG / 'manifest/local_split_v1.json'), verify_splits=('test',))


def load_gt(labels_dir, sample_id, width, height):
    gt = []
    lf = Path(labels_dir) / (sample_id + '.txt')
    if lf.exists():
        for line in lf.read_text(encoding='utf-8-sig').splitlines():
            if not line.strip():
                continue
            _, x, y, w, h = map(float, line.split())
            gt.append([(x - w / 2) * width, (y - h / 2) * height,
                       (x + w / 2) * width, (y + h / 2) * height])
    return gt


def froc_at_budgets(rows, budgets):
    n_img = len(rows)
    total_small_gt = sum(sum(1 for b in r['gt'] if (b[2] - b[0]) * (b[3] - b[1]) < 1024) for r in rows)
    if total_small_gt == 0:
        return {b: None for b in budgets}
    all_scores = sorted(set(s for r in rows for s in r['confidence']), reverse=True)
    if not all_scores:
        return {b: 0.0 for b in budgets}
    if len(all_scores) > 200:
        idx = np.linspace(0, len(all_scores) - 1, 200, dtype=int)
        thresholds = [all_scores[i] for i in idx]
    else:
        thresholds = all_scores
    results = {b: 0.0 for b in budgets}
    for thr in thresholds:
        fp = tp_small = 0
        for r in rows:
            sel = [i for i, s in enumerate(r['confidence']) if s >= thr]
            if not sel and not r['gt']:
                continue
            matches = match_frame(r['gt'], [r['boxes_xyxy'][i] for i in sel],
                                  [r['confidence'][i] for i in sel])
            fp += sum(m['fp'] for m in matches)
            tp_small += sum(m['small_tp'] for m in matches)
        fp_per_1000 = fp * 1000 / n_img
        recall_small = tp_small / total_small_gt
        for b in budgets:
            if fp_per_1000 <= b and recall_small > results[b]:
                results[b] = recall_small
    return results


def eval_model(tag, det_path, img_dir, lbl_dir):
    model = load_detector(str(det_path))
    images = sorted(Path(img_dir).glob('*.png'))
    if not images:
        print(f'  SKIP {tag}: no images at {img_dir}', flush=True)
        return None
    rows = list(_results(model, images, batch=8, imgsz=640, device='cuda', conf=.001))
    for row in rows:
        row['gt'] = load_gt(lbl_dir, row['sample_id'], row['width'], row['height'])
    froc = froc_at_budgets(rows, BUDGETS)
    del model
    torch.cuda.empty_cache()
    return {'run': tag, 'froc': froc}


def variant_runs(variant, seeds):
    out = []
    for f in seeds:
        for d in '012':
            tag = f'{variant}_f{f}_d{d}'
            det = RUNS / 'detection' / tag / 'weights' / 'best.pt'
            exp = RUNS / 'fusion' / f'{variant}_s{f}' / 'export_test'
            if det.exists() and (exp / 'images' / 'test').exists():
                out.append((tag, det, exp))
            else:
                print(f'  SKIP {tag} (missing)', flush=True)
    return out


def baseline_runs(name):
    out = []
    for d in '012':
        tag = f'{name}_f0_d{d}'
        det = RUNS / 'detection' / tag / 'weights' / 'best.pt'
        exp = RUNS / 'fusion' / f'{name}_s0' / 'export_test'
        if det.exists() and (exp / 'images' / 'test').exists():
            out.append((tag, det, exp))
        else:
            print(f'  SKIP {tag} (missing)', flush=True)
    return out


print('FROC_P3_START', flush=True)
all_results = {}
jobs = []
jobs += variant_runs('noattn', ('3407', '3408', '3409'))
jobs += variant_runs('nodfu', ('3407', '3408', '3409'))
jobs += baseline_runs('cddfuse')
jobs += baseline_runs('swinfusion')
for tag, det, exp in jobs:
    print(f'  {tag}...', flush=True)
    r = eval_model(tag, det, exp / 'images' / 'test', exp / 'labels' / 'test')
    if r:
        all_results[tag] = r
        print(f"    @50: {r['froc'][50]:.4f}", flush=True)

out_file = OUT / 'froc_budget_p3.json'
existing = json.loads(out_file.read_text()) if out_file.exists() else {}
existing.update(all_results)
out_file.write_text(json.dumps(existing, indent=2))
print(f'FROC_P3_DONE: {len(all_results)} runs evaluated -> {out_file}', flush=True)
