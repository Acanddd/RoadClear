from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseEnhanceModel(ABC):
    """
    视频/图像增强模型基类。

    约定输入输出均为单帧图像的 NumPy 数组，形状为 (H, W, C)，
    通道顺序采用 OpenCV 默认的 BGR。
    """

    @abstractmethod
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧图像进行增强。

        :param frame: 输入帧，BGR 格式，uint8 类型。
        :return: 增强后的帧，BGR 格式，uint8 类型。
        """
        raise NotImplementedError

    def to(self, device: Any) -> "BaseEnhanceModel":
        """
        可选的设备迁移接口，默认直接返回自身。
        具体模型（如基于 PyTorch）可以覆写该方法。
        """
        return self
