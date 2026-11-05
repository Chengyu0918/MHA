"""Export fused images from an external pretrained baseline (P3-C).

Same loop/layout/quantization as export_variant.py / export_legacy_01.py:
reads the project manifest, fuses every included pair via pkd.runtime.fuse_pair
(RGB->Y, fuse Y, restore VI chroma), writes images/labels/data.yaml so the
standard train_detector + adapter_eval pipeline runs unchanged.
"""
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
from baseline_wrappers import build_baseline  # noqa: E402


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main(method, output_dir, splits, manifest_path, device='cuda'):
    model = build_baseline(method, device)
    model.eval()
    manifest = load_manifest(manifest_path, verify_splits=splits)
    output_dir = Path(output_dir).absolute().resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    ckpt_files = {
        'cddfuse': ['/tmp/p3c_repos/CDDFuse/models/CDDFuse_IVF.pth'],
        'swinfusion': ['/tmp/p3c_repos/SwinFusion/Model/Infrared_Visible_Fusion/'
                       'Infrared_Visible_Fusion/models/10000_E.pth'],
    }[method]
    metadata = {'state': 'running',
                'adapter': 'export_baseline (external pretrained; no production code modified)',
                'method': method,
                'source': 'official repo pretrained checkpoint, unmodified',
                'checkpoint_sha256': {p: sha256_file(p) for p in ckpt_files},
                'data_sha256': manifest['data_sha256'],
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
                    print(method, split, index + 1, '/', len(dataset.samples), flush=True)
    lines = ['path: ' + json.dumps(str(output_dir).replace('\\', '/'))]
    lines += [s + ': images/' + s for s in splits]
    lines += ['names:', '  0: drone']
    (output_dir / 'data.yaml').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    metadata['state'] = 'completed'
    atomic_json(output_dir / 'export_manifest.json', metadata)
    print('BASELINE_EXPORT_DONE', method, output_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], tuple(sys.argv[3].split(',')), sys.argv[4])
