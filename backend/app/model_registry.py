"""Validated model availability shared by routes and the dispatcher."""
from time import perf_counter
import numpy as np
from . import settings

instances = {}
statuses = {}

def initialize():
    from .models.aodnet import AODNetEnhancer
    from .models.prenet import PreNetEnhancer
    from .models.transweather import TransWeatherEnhancer
    from .scheduler.weather_classifier import WeatherCNNClassifier
    from .evaluation.detector import get_detector, get_license_plate_detector
    factories = {
        'aodnet': lambda d: AODNetEnhancer(device=d, use_onnx=True, onnx_path=str(settings.model_path('aodnet'))),
        'prenet': lambda d: PreNetEnhancer(device=d, use_onnx=True, onnx_path=str(settings.model_path('prenet'))),
        'transweather': lambda d: TransWeatherEnhancer(device=d),
        'weather': lambda d: WeatherCNNClassifier(device=d, model_path=str(settings.model_path('weather'))),
        'vehicle': lambda d: get_detector(device=d),
        'plate': lambda d: get_license_plate_detector(device=d),
    }
    frame = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    for name, factory in factories.items():
        status = {'available': False, 'state': 'loading', 'reason': None, 'device': settings.DEVICE,
                  'file': settings.MODEL_FILES[name], 'provenance': 'unverified original training/export provenance'}
        statuses[name] = status
        start = perf_counter()
        try:
            if name in settings.DISABLED_MODELS:
                status['state'] = 'disabled'
                raise RuntimeError('Disabled by ROADCLEAR_DISABLED_MODELS')
            if not settings.model_path(name).is_file():
                status['state'] = 'missing'
                raise FileNotFoundError(f'Missing model: {settings.MODEL_FILES[name]}')
            d = settings.device()
            instance = factory(d)
            if name in ('aodnet', 'prenet', 'transweather'):
                output = instance.enhance(frame)
                if output.shape != frame.shape or output.dtype != np.uint8 or not np.isfinite(output).all():
                    raise RuntimeError('Invalid model output')
            elif name == 'weather':
                prediction = instance.predict(frame)
                if not np.isfinite(prediction.probs).all():
                    raise RuntimeError('Invalid classifier probabilities')
            else:
                instance.detect(frame)
            instances[name] = instance
            status.update(available=True, state='available', device=d)
        except Exception as exc:
            status['reason'] = f'{type(exc).__name__}: {exc}'
            if status['state'] == 'loading':
                status['state'] = 'load_failed'
        status['load_and_smoke_seconds'] = round(perf_counter() - start, 4)

def require(name):
    name = 'transweather' if name == 'hdcwnet' else name
    if name not in instances:
        raise RuntimeError(f'{name} unavailable: {statuses.get(name, {}).get("reason", "not loaded")}')
    return instances[name]

def capabilities():
    result = {key: dict(value) for key, value in statuses.items()}
    deps = ('weather', 'aodnet', 'prenet', 'transweather')
    missing = [key for key in deps if not result.get(key, {}).get('available')]
    result['auto'] = {'available': not missing, 'reason': 'Unavailable: ' + ', '.join(missing) if missing else None}
    result['simple'] = {'available': True, 'reason': None}
    return result
