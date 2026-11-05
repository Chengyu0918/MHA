"""Train one structural-ablation variant via the production runtime.

Usage: python /tmp/train_variant.py --variant noattn --config CFG --run-dir DIR
Swaps pkd.fusion.PKDFusion before importing pkd.runtime so runtime's
function-level import resolves to the variant class. Writes a variant
manifest into the run directory for provenance transparency.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

CODE = '/home/stu5/chengyu0918/LESAFusion_TAES_20261028/code'
sys.path.insert(0, CODE)
sys.path.insert(0, '/tmp')
import pkd.fusion as fusion_mod
from pkd_variants import VARIANTS


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--variant', required=True, choices=sorted(VARIANTS))
    ap.add_argument('--config', required=True)
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--epochs', type=int, default=None)
    ap.add_argument('--max-train-batches', type=int, default=0)
    ap.add_argument('--max-val-batches', type=int, default=0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    variant_cls = VARIANTS[args.variant]
    fusion_mod.PKDFusion = variant_cls
    from pkd.runtime import train
    train(args.config, args.run_dir, device=args.device, epochs=args.epochs,
          max_train_batches=args.max_train_batches, max_val_batches=args.max_val_batches,
          resume=args.resume)
    manifest = {'variant': args.variant,
                'variant_class': variant_cls.__name__,
                'variant_module': '/tmp/pkd_variants.py',
                'variant_module_sha256': sha('/tmp/pkd_variants.py'),
                'base_config': str(Path(args.config).resolve()),
                'base_config_sha256': sha(args.config),
                'production_fusion_sha256': sha(Path(CODE) / 'pkd' / 'fusion.py'),
                'note': 'production pkd untouched; PKDFusion symbol swapped at runtime'}
    out = Path(args.run_dir) / 'variant_manifest.json'
    out.write_text(json.dumps(manifest, indent=2))
    print('VARIANT_DONE', args.variant, args.run_dir)


if __name__ == '__main__':
    main()
