from typing import Dict, List

import cv2
import numpy as np

from .detector import DetectionResult


def compute_mean_confidence(detections: List[List[DetectionResult]]) -> float:
    """所有帧检测结果的平均置信度。"""
    scores = [d.score for frame_dets in detections for d in frame_dets]
    if not scores:
        return 0.0
    return float(np.mean(scores))


def compute_high_confidence_count(
    detections: List[List[DetectionResult]], threshold: float = 0.7
) -> int:
    """所有帧中置信度高于阈值的检测总数。"""
    return sum(
        1
        for frame_dets in detections
        for d in frame_dets
        if d.score >= threshold
    )


def compute_detection_count_change_rate(
    base_dets: List[List[DetectionResult]],
    enhanced_dets: List[List[DetectionResult]],
) -> float:
    """
    检测数量变化率: (增强后总检测数 - 增强前总检测数) / 增强前总检测数。
    正值表示增强后检测到更多目标，负值表示减少。
    """
    base_total = sum(len(frame) for frame in base_dets)
    enh_total = sum(len(frame) for frame in enhanced_dets)
    if base_total == 0:
        return float(enh_total) if enh_total > 0 else 0.0
    return (enh_total - base_total) / base_total


def compute_contrast_gain(
    base_frames: List[np.ndarray],
    enhanced_frames: List[np.ndarray],
) -> float:
    """
    图像对比度增益（基于 Michelson 对比度）。
    Michelson = (Imax - Imin) / (Imax + Imin)
    返回增强后对比度相对于增强前的平均增益比。
    """
    if not base_frames or not enhanced_frames:
        return 0.0

    n = min(len(base_frames), len(enhanced_frames))
    gains: List[float] = []

    for i in range(n):
        base_gray = cv2.cvtColor(base_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)
        enh_gray = cv2.cvtColor(enhanced_frames[i], cv2.COLOR_BGR2GRAY).astype(np.float64)

        def _michelson(img: np.ndarray) -> float:
            i_max = img.max()
            i_min = img.min()
            denom = i_max + i_min
            if denom < 1e-6:
                return 0.0
            return (i_max - i_min) / denom

        mc_base = _michelson(base_gray)
        mc_enh = _michelson(enh_gray)

        if mc_base > 1e-6:
            gains.append((mc_enh - mc_base) / mc_base)
        else:
            gains.append(mc_enh)

    return float(np.mean(gains)) if gains else 0.0


def compute_noise_suppression_rate(
    base_frames: List[np.ndarray],
    enhanced_frames: List[np.ndarray],
) -> float:
    """
    噪声抑制率（基于拉普拉斯方差）。
    拉普拉斯方差越高表示高频细节/噪声越多。
    抑制率 = (base_var - enh_var) / base_var，正值表示噪声减少。
    """
    if not base_frames or not enhanced_frames:
        return 0.0

    n = min(len(base_frames), len(enhanced_frames))
    rates: List[float] = []

    for i in range(n):
        base_gray = cv2.cvtColor(base_frames[i], cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(enhanced_frames[i], cv2.COLOR_BGR2GRAY)

        base_var = cv2.Laplacian(base_gray, cv2.CV_64F).var()
        enh_var = cv2.Laplacian(enh_gray, cv2.CV_64F).var()

        if base_var > 1e-6:
            rates.append((base_var - enh_var) / base_var)
        else:
            rates.append(0.0)

    return float(np.mean(rates)) if rates else 0.0


def compute_temporal_stability(detections: List[List[DetectionResult]]) -> float:
    """
    时序稳定性：逐帧检测数量的标准差归一化值。
    值越小说明检测越稳定。返回稳定性得分 = 1 / (1 + 归一化标准差)。
    """
    if len(detections) < 2:
        return 1.0

    counts = [len(frame) for frame in detections]
    mean_count = np.mean(counts)
    std_count = np.std(counts)

    if mean_count < 1e-6:
        return 1.0

    normalized_std = std_count / mean_count
    return float(1.0 / (1.0 + normalized_std))
