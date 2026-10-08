"""TransWeather ONNX adapter; original checkpoint provenance remains unverified."""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from ..settings import model_path
from .base_model import BaseEnhanceModel


class TransWeatherEnhancer(BaseEnhanceModel):
    def __init__(self, device='cpu', onnx_path=None):
        path = Path(onnx_path or model_path('transweather'))
        if not path.is_file():
            raise FileNotFoundError(str(path))
        providers = ['CPUExecutionProvider']
        if str(device).startswith('cuda'):
            if 'CUDAExecutionProvider' not in ort.get_available_providers():
                raise RuntimeError('CUDA ONNX provider unavailable')
            providers.insert(0, 'CUDAExecutionProvider')
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(path), options, providers=providers)
        self.input = self.session.get_inputs()[0]
        self.device = device

    def to(self, device):
        if str(device) != str(self.device):
            raise ValueError('Restart with the requested device to recreate ONNX sessions')
        return self

    def enhance(self, frame):
        h, w = frame.shape[:2]
        shape = self.input.shape
        # Retained export fails on 96x128; use square multiples of 64, then restore original size.
        side = min(256, ((max(h, w) + 63) // 64) * 64)
        ih = shape[2] if isinstance(shape[2], int) else side
        iw = shape[3] if isinstance(shape[3], int) else side
        rgb = cv2.cvtColor(cv2.resize(frame, (iw, ih)), cv2.COLOR_BGR2RGB)
        x = ((rgb.astype(np.float32) / 255.0 - .5) / .5).transpose(2, 0, 1)[None]
        y = self.session.run(None, {self.input.name: x})[0]
        if y.ndim != 4 or y.shape[0] != 1 or y.shape[1] != 3 or not np.isfinite(y).all():
            raise RuntimeError('TransWeather returned invalid output')
        # Matches the retained original PyTorch adapter normalization.
        rgb = np.clip((y[0].transpose(1, 2, 0) * .5 + .5) * 255, 0, 255).astype(np.uint8)
        return cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (w, h))
