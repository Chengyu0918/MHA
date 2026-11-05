"""Export fused images from a structural-ablation variant checkpoint.

Same loop/quantization/layout as export_legacy_01.py, but rebuilds the model
from the run's variant manifest so the variant forward path is used. No
production code is modified.
"""
import io
import json
import shutil
import sys
import hashlib
from pathlib import Path
import torch

CODE = Path('/home/stu5/chengyu0918/LESAFusion_TAES_20261028/code')
sys.path.insert(0, str(CODE))
sys.path.insert(0, '/tmp')
from pkd.data import load_manifest, PairedDataset  # noqa: E402
from pkd.runtime import _save_image, atomic_json, fuse_pair, source_hashes  # noqa: E402
from pkd_variants import VARIANTS  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(variant, checkpoint_path, output_dir, splits, manifest_path, device='cuda'):
    content = Path(checkpoint_path).read_bytes()
    checkpoint = torch.load(io.BytesIO(content), map_location='cpu')
    provenance = checkpoint['provenance']
    model_kwargs = dict(provenance.get("model_kwargs", provenance.get("resolved_config", {}).get("model_kwargs", {})))
    model = VARIANTS[variant](**model_kwargs)
    model.load_state_dict(checkpoint['model'], strict=True)
    model = model.to(device).float().eval()
    manifest = load_manifest(manifest_path, verify_splits=splits)
    assert provenance['data_sha256'] == manifest['data_sha256'], 'manifest mismatch'
    output_dir = Path(output_dir).absolute().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    metadata = {'state': 'running',
                'adapter': 'export_variant (variant forward; no production code modified)',
                'variant': variant,
                'version_id': provenance['version_id'],
                'data_sha256': manifest['data_sha256'],
                'checkpoint_sha256': hashlib.sha256(content).hexdigest(),
                'checkpoint_epoch': checkpoint['epoch'],
                'source_sha256': source_hashes(),
                'splits': list(splits), 'tile_size': 0, 'overlap': 32,
                'allow_test': 'test' in splits,
                'quantization': 'same RGB round-to-nearest uint8; native_dimensions True',
                'model_inputs': ['vi_y', 'ir_y'],
                'expected_ids': {s: sorted(r['sample_id'] for r in manifest['samples']
                                           if r['included'] and r['split'] == s) for s in splits},
                'files': {}}
    atomic_json(output_dir / 'export_manifest.json', metadata)
    with torch.no_grad():
        for split in splits:
            dataset = PairedDataset(manifest, split, full_frame=True, allow_test='test' in splits)
            (output_dir / 'images' / split).mkdir(parents=True)
            (output_dir / 'labels' / split).mkdir(parents=True)
            for index, record in enumerate(dataset.samples):
                sample = dataset[index]
                result = fuse_pair(model,
                                   sample['vi_rgb'].unsqueeze(0).to(device),
                                   sample['ir_rgb'].unsqueeze(0).to(device), 0, 32)
                name = record['sample_id']
                image_path = output_dir / 'images' / split / (name + '.png')
                label_path = output_dir / 'labels' / split / (name + '.txt')
                _save_image(result, image_path)
                side = 'ir' if record['label_source'] == 'ir_repair' else 'eo'
                shutil.copyfile(record[side + '_label_path'], label_path)
                metadata['files'][split + '/' + name] = {
                    'image_sha256': sha256_file(image_path),
                    'label_sha256': sha256_file(label_path), 'source_label_side': side,
                    'sample_id': name, 'source_eo_sha256': record['eo_sha256'],
                    'source_ir_sha256': record['ir_sha256'],
                    'width': record['width'], 'height': record['height']}
                if index % 200 == 0:
                    print(split, index + 1, '/', len(dataset.samples), flush=True)
    lines = ['path: ' + json.dumps(str(output_dir).replace('\\', '/'))]
    lines += [s + ': images/' + s for s in splits]
    lines += ['names:', '  0: drone']
    (output_dir / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    metadata['state'] = 'completed'
    atomic_json(output_dir / 'export_manifest.json', metadata)
    print('VARIANT_EXPORT_DONE', variant, output_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], tuple(sys.argv[4].split(',')), sys.argv[5])
