"""Aggregate the MHA 5x3 matrix (fusion seeds 3407-3411 x det seeds 0-2).

Reads runs_mha_followup/eval/h{1,4}_f{seed}_d{det}.json, builds R_small
tables, means/SDs, per-det-seed h4-h1 diffs, and leave-one-fusion-seed-out
sensitivity. Tolerates missing cells (prints partial matrix state).
"""
import json
from pathlib import Path
from statistics import mean, stdev

EVAL = Path('/home/stu5/chengyu0918/LESAFusion_TAES_20261028/runs_mha_followup/eval')
SEEDS = [3407, 3408, 3409, 3410, 3411]
DETS = [0, 1, 2]

FIELDS = {'R_small': 'small_recall', 'R_all': 'recall', 'FP1000': 'fp_per_1000', 'AP50': None}


def load_cell(h, f, d):
    p = EVAL / f'h{h}_f{f}_d{d}.json'
    if not p.exists():
        return None
    j = json.loads(p.read_text())
    op = j['test_operating_point']
    return {'R_small': op.get('small_recall'), 'R_all': op.get('recall'),
            'FP1000': op.get('fp_per_1000'), 'AP50': j.get('test_AP50'),
            'thr': j.get('frozen_threshold')}


def cells(h, field):
    return {(f, d): load_cell(h, f, d)[field]
            for f in SEEDS for d in DETS if load_cell(h, f, d)}


def fmt(v):
    return f'{v:.4f}' if v is not None else '  --  '


def print_table(h, field):
    data = cells(h, field)
    print(f'\n## h{h} {field}')
    print('| fusion\\det | ' + ' | '.join(f'det{d}' for d in DETS) + ' | 行均值 |')
    print('|---' * (len(DETS) + 2) + '|')
    rmeans = []
    for f in SEEDS:
        row = [data.get((f, d)) for d in DETS]
        vals = [v for v in row if v is not None]
        rmeans.append(mean(vals) if vals else None)
        rm = f'{mean(vals):.4f}' if len(vals) == len(DETS) else (f'({mean(vals):.4f})' if vals else '--')
        print(f'| f{f} | ' + ' | '.join(fmt(v) for v in row) + f' | {rm} |')
    colmeans = []
    for d in DETS:
        vals = [data.get((f, d)) for f in SEEDS if data.get((f, d)) is not None]
        colmeans.append(mean(vals) if vals else None)
    allv = [v for v in rmeans if v is not None]
    tail = f'**{mean(allv):.4f}**' if allv else '--'
    print('| **均值** | ' + ' | '.join(f'**{v:.4f}**' if v is not None else '--' for v in colmeans)
          + f' | {tail} |')
    flat = [v for v in data.values() if v is not None]
    if len(flat) > 1:
        print(f'\n全格子 SD={stdev(flat):.4f} n={len(flat)} 均值={mean(flat):.4f}')


def main():
    print_table(1, 'R_small')
    print_table(4, 'R_small')
    print('\n## h4 − h1 (R_small)')
    for d in DETS:
        v1 = [load_cell(1, f, d)['R_small'] for f in SEEDS if load_cell(1, f, d)]
        v4 = [load_cell(4, f, d)['R_small'] for f in SEEDS if load_cell(4, f, d)]
        n = min(len(v1), len(v4))
        if n:
            print(f'det{d}: h4={mean(v4):.4f} h1={mean(v1):.4f} diff={mean(v4)-mean(v1):+.4f} (n={n})')
    print('\n## LOO 敏感性 (剔除单个融合种子)')
    for k in SEEDS:
        v1 = [load_cell(1, f, d)['R_small'] for f in SEEDS if f != k for d in DETS if load_cell(1, f, d)]
        v4 = [load_cell(4, f, d)['R_small'] for f in SEEDS if f != k for d in DETS if load_cell(4, f, d)]
        if v1 and v4:
            print(f'剔除 f{k}: h4={mean(v4):.4f} h1={mean(v1):.4f} diff={mean(v4)-mean(v1):+.4f}')
    print('\n## AP50')
    print_table(1, 'AP50')
    print_table(4, 'AP50')
    miss = [(h, f, d) for h in (1, 4) for f in SEEDS for d in DETS if not load_cell(h, f, d)]
    print(f'\n缺失格子: {len(miss)} {miss if miss else ""}')


if __name__ == '__main__':
    main()
