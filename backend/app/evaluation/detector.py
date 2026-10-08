import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from ..settings import model_path


# 项目根目录路径
_BACKEND_DIR = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = model_path("vehicle")
FALLBACK_WEIGHTS = _BACKEND_DIR / "yolov5su.pt"
LICENSE_PLATE_WEIGHTS = model_path("plate")


@dataclass
class DetectionResult:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    cls: int
    cls_name: str  # 增加类别名称支持
    score: float


class YOLOv5Detector:
    """
    基于自定义训练权重的交通目标检测封装。
    """

    def __init__(self, weights_path: str | Path | None = None, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        weights = Path(weights_path) if weights_path else DEFAULT_WEIGHTS

        if not weights.is_file():
            raise FileNotFoundError(str(weights))

        print(f"[YOLOv5] Loading custom weights from: {weights}")
        self.model = YOLO(str(weights))
        self.model.to(self.device)
        self.model.model.eval()
        self.names = self.model.names  # 获取类别映射表

    @torch.no_grad()
    def detect(self, frame: np.ndarray, conf_thres: float = 0.25) -> List[DetectionResult]:
        """
        对单帧 BGR 图像进行目标检测。
        """
        results = self.model.predict(frame, conf=conf_thres, verbose=False, device=str(self.device))
        detections: List[DetectionResult] = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                detections.append(
                    DetectionResult(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        cls=cls,
                        cls_name=self.names.get(cls, str(cls)),
                        score=conf,
                    )
                )
        return detections


class LicensePlateDetector:
    """
    基于CCPD数据集训练的车牌检测器（YOLOv8）。
    """

    def __init__(self, weights_path: str | Path | None = None, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        weights = Path(weights_path) if weights_path else LICENSE_PLATE_WEIGHTS

        if not weights.exists():
            print(f"[LicensePlate] Weights not found at: {weights}")
            raise FileNotFoundError(f"License plate weights file not found: {weights}")

        print(f"[LicensePlate] Loading CCPD weights from: {weights}")
        self.model = YOLO(str(weights))
        self.model.to(self.device)
        self.model.model.eval()
        self.names = self.model.names  # 获取类别映射表

    @torch.no_grad()
    def detect(self, frame: np.ndarray, conf_thres: float = 0.25) -> List[DetectionResult]:
        """
        对单帧 BGR 图像进行车牌检测。
        """
        results = self.model.predict(frame, conf=conf_thres, verbose=False, device=str(self.device))
        detections: List[DetectionResult] = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())
                detections.append(
                    DetectionResult(
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        cls=cls,
                        cls_name=self.names.get(cls, str(cls)),
                        score=conf,
                    )
                )
        return detections


# 全局检测器实例，延迟加载（首次调用时初始化）
_detector_instance = None
_lp_detector_instance = None


def get_detector(device: str = "cpu") -> YOLOv5Detector:
    """
    获取全局车辆检测器实例（延迟加载）。
    避免启动时因网络问题导致服务无法启动。
    """
    global _detector_instance
    if _detector_instance is None:
        print("[YOLOv5] Initializing vehicle detector (first call)...")
        _detector_instance = YOLOv5Detector(device=device)
    return _detector_instance


def get_license_plate_detector(device: str = "cpu") -> LicensePlateDetector:
    """
    获取全局车牌检测器实例（延迟加载）。
    """
    global _lp_detector_instance
    if _lp_detector_instance is None:
        print("[LicensePlate] Initializing license plate detector (first call)...")
        _lp_detector_instance = LicensePlateDetector(device=device)
    return _lp_detector_instance


# 保持向后兼容的别名（首次访问时初始化）
class _LazyDetector:
    """延迟加载的检测器代理类"""
    def __getattr__(self, name):
        return getattr(get_detector(), name)


detector = _LazyDetector()
