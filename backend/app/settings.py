"""Filesystem and deployment configuration, independent of the working directory."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / '.env', override=False)

def path_env(name, default):
    path = Path(os.getenv(name, str(default))).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()

MODEL_DIR = path_env('ROADCLEAR_MODEL_DIR', ROOT / 'models')
DATA_DIR = path_env('ROADCLEAR_DATA_DIR', ROOT / 'runtime')
CONFIG_PATH = path_env('ROADCLEAR_CONFIG_PATH', DATA_DIR / 'config/parameters.json')
DEVICE = os.getenv('ROADCLEAR_DEVICE', 'cpu')
MAX_UPLOAD_BYTES = int(os.getenv('ROADCLEAR_MAX_UPLOAD_MB', '100')) * 1024 * 1024
CORS_ORIGINS = os.getenv('ROADCLEAR_CORS_ORIGINS', 'http://localhost:5173,http://127.0.0.1:5173').split(',')
DEMO_DIR = path_env('ROADCLEAR_DEMO_DIR', ROOT / 'frontend/image_display')
SHOWCASE_DIR = path_env('ROADCLEAR_SHOWCASE_DIR', ROOT / 'frontend/vue/public/algorithm-showcase')
DISABLED_MODELS = {value.strip().lower() for value in os.getenv('ROADCLEAR_DISABLED_MODELS', '').split(',') if value.strip()}
MODEL_FILES = {
    'aodnet': 'aodnet.onnx', 'prenet': 'prenet.onnx',
    'transweather': 'transweather_desnow.onnx',
    'weather': 'weather_classifier.pth', 'vehicle': 'vehicle.pt', 'plate': 'plate.pt',
}

def model_path(name):
    return MODEL_DIR / MODEL_FILES[name]

def device():
    import torch
    if DEVICE == 'cpu':
        return 'cpu'
    if DEVICE == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    raise RuntimeError(f'Requested device {DEVICE!r} is unavailable; use ROADCLEAR_DEVICE=cpu')

os.environ.setdefault('YOLO_CONFIG_DIR', str(DATA_DIR / 'ultralytics'))
os.environ.setdefault('MPLCONFIGDIR', str(DATA_DIR / 'matplotlib'))

os.environ.setdefault("MPLBACKEND", "Agg")
