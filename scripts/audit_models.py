"""Persist model checksums, ONNX interfaces and actual adapter smoke results."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from app import settings, model_registry
import cv2
import numpy as np
import onnxruntime as ort
from time import perf_counter

def main():
    model_registry.initialize()
    records = json.loads((ROOT / 'docs/model-manifest.json').read_text(encoding='utf-8'))
    for record in records:
        name = record['id']
        path = settings.model_path(name)
        actual_hash = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest() if path.is_file() else None
        record['checksum_matches'] = actual_hash == record['sha256']
        record['validation'] = model_registry.statuses[name]
        if path.suffix == '.onnx' and path.is_file():
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            session = ort.InferenceSession(str(path), options, providers=['CPUExecutionProvider'])
            record['onnx_interface'] = {
                'inputs': [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in session.get_inputs()],
                'outputs': [{'name': v.name, 'shape': v.shape, 'type': v.type} for v in session.get_outputs()],
            }
        if name in ('aodnet', 'prenet', 'transweather') and record['validation']['available']:
            weather = {'aodnet':'fog','prenet':'rain','transweather':'snow'}[name]
            source = cv2.imread(str(ROOT / 'frontend/image_display' / f'{weather}1.jpg'))
            frame = cv2.resize(source, (128,96))
            start = perf_counter()
            output = model_registry.instances[name].enhance(frame)
            record['sample_inference'] = {'input_shape': list(frame.shape), 'output_shape': list(output.shape),
                'dtype': str(output.dtype), 'finite': bool(np.isfinite(output).all()),
                'seconds': round(perf_counter()-start, 4), 'device': 'cpu',
                'output_min': int(output.min()), 'output_max': int(output.max()),
                'mean_absolute_pixel_change': float(np.abs(output.astype(float)-frame).mean())}
        elif name == 'weather':
            record['interface'] = 'BGR uint8 frame -> fog/rain/snow softmax probabilities'
        else:
            record['interface'] = 'BGR uint8 frame -> bounding boxes, classes and confidence; not OCR'
    (ROOT / 'docs/model-manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print('Wrote docs/model-manifest.json')

if __name__ == '__main__':
    main()
