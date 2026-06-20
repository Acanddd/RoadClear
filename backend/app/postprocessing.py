import cv2
import numpy as np

from .config_manager import ParameterManager

# 延迟初始化参数管理器
_parameter_manager = None

def get_parameter_manager() -> ParameterManager:
    """获取参数管理器实例（延迟初始化）"""
    global _parameter_manager
    if _parameter_manager is None:
        _parameter_manager = ParameterManager()
    return _parameter_manager


def apply_clahe(frame: np.ndarray, clip_limit: float = 2.0, tile_grid_size=(8, 8)) -> np.ndarray:
    """使用 CLAHE 提升图像局部对比度。"""
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l = clahe.apply(l)
    merged = cv2.merge([l, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(
    frame: np.ndarray,
    kernel_size=(9, 9),
    sigma: float = 10.0,
    amount: float = 1.0,
    threshold: int = 0,
) -> np.ndarray:
    """应用非锐化掩膜增强边缘。"""
    blurred = cv2.GaussianBlur(frame, kernel_size, sigma)
    sharpened = cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)

    if threshold > 0:
        low_contrast_mask = np.abs(frame.astype(np.int16) - blurred.astype(np.int16)) < threshold
        sharpened = np.where(low_contrast_mask, frame, sharpened)

    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_fast_denoise(
    frame: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 50.0,
    sigma_space: float = 50.0,
) -> np.ndarray:
    """使用双边滤波进行快速降噪，速度显著优于 NL-Means。"""
    return cv2.bilateralFilter(frame, diameter, sigma_color, sigma_space)


def apply_dynamic_range_compression(
    frame: np.ndarray,
    bright_threshold: int = 220,
    slope: float = 0.5,
) -> np.ndarray:
    """压制过曝区域，避免车灯等高亮区域过于刺眼。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    v = hsv[:, :, 2]
    mask = v > bright_threshold
    if np.any(mask):
        v[mask] = bright_threshold + (v[mask] - bright_threshold) * slope
        hsv[:, :, 2] = np.clip(v, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return frame


def postprocess_frame(
    frame: np.ndarray,
    use_denoise: bool = None,
    use_dynamic_range: bool = None,
) -> np.ndarray:
    """对增强结果执行后处理：CLAHE、Unsharp Mask、可选快速降噪与动态范围压缩。"""
    config = get_parameter_manager().get_config().postprocessing

    output = frame.copy()

    # CLAHE
    if config.clahe.enabled:
        output = apply_clahe(
            output,
            clip_limit=config.clahe.clip_limit,
            tile_grid_size=config.clahe.tile_grid_size
        )

    # Unsharp Mask
    if config.unsharp_mask.enabled:
        output = apply_unsharp_mask(
            output,
            kernel_size=config.unsharp_mask.kernel_size,
            sigma=config.unsharp_mask.sigma,
            amount=config.unsharp_mask.amount,
            threshold=config.unsharp_mask.threshold
        )

    # Denoise
    if use_denoise is None:
        use_denoise = config.denoise.enabled
    if use_denoise:
        output = apply_fast_denoise(
            output,
            diameter=config.denoise.diameter,
            sigma_color=config.denoise.sigma_color,
            sigma_space=config.denoise.sigma_space
        )

    # Dynamic Range Compression
    if use_dynamic_range is None:
        use_dynamic_range = config.dynamic_range.enabled
    if use_dynamic_range:
        output = apply_dynamic_range_compression(
            output,
            bright_threshold=config.dynamic_range.bright_threshold,
            slope=config.dynamic_range.slope
        )

    return output
