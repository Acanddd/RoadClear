import os
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from .base_model import BaseEnhanceModel

# 路径配置
_BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
TRANSWEATHER_WEIGHTS_PATH = str(_BACKEND_DIR.parent / "TransWeather-main" / "epoch_90")


class HDCWNetEnhancer(BaseEnhanceModel):
    """
    HDCWNet 去雪模型封装 (使用 TransWeather)
    
    处理流程:
    1. TransWeather 去雪 (PyTorch模式)
    
    保留原有的 HDCWNet 接口名称以保持兼容性
    """

    def __init__(
        self, 
        device: str = "cpu",
        weights_path: str | None = None,
        transweather_path: str | None = None
    ) -> None:
        """
        初始化去雪模型
        
        Args:
            device: 设备 ('cpu' 或 'cuda')
            weights_path: 保留参数，为了兼容性（不使用）
            transweather_path: TransWeather 权重文件路径
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        # 默认使用预设路径
        if transweather_path is None:
            transweather_path = TRANSWEATHER_WEIGHTS_PATH
        
        # 检查文件是否存在
        if not os.path.exists(transweather_path):
            raise FileNotFoundError(f"TransWeather weights not found at: {transweather_path}")
        
        print(f"[HDCWNetEnhancer] Using device: {self.device}")
        
        # 加载 TransWeather 去雪模型 (PyTorch)
        print("[HDCWNetEnhancer] Loading TransWeather desnow model (PyTorch)...")
        self._load_transweather(transweather_path)
        
        print("[HDCWNetEnhancer] Model initialization complete!")
        print("[HDCWNetEnhancer] Pipeline: TransWeather (desnow)")

    def _load_transweather(self, weights_path: str):
        """加载TransWeather模型"""
        # 动态导入TransWeather模型
        import sys
        # 获取 TransWeather-main 目录的绝对路径
        transweather_dir = str(Path(weights_path).parent)
        if transweather_dir not in sys.path:
            sys.path.insert(0, transweather_dir)
        
        try:
            from transweather_model import Transweather
        except ImportError as e:
            # 如果导入失败，尝试从父目录导入
            transweather_parent = str(Path(weights_path).parent.parent)
            if transweather_parent not in sys.path:
                sys.path.insert(0, transweather_parent)
            from transweather_model import Transweather
        
        # 创建模型
        self.transweather = Transweather()
        self.transweather = self.transweather.to(self.device)
        
        # 加载权重
        state_dict = torch.load(weights_path, map_location=self.device, weights_only=False)
        
        # 处理DataParallel保存的权重
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # 移除'module.'前缀
            else:
                name = k
            new_state_dict[name] = v
        
        self.transweather.load_state_dict(new_state_dict)
        self.transweather.eval()
        
        print(f"[HDCWNetEnhancer] TransWeather weights loaded from: {weights_path}")

    def to(self, device: Any) -> "HDCWNetEnhancer":
        """设备迁移"""
        self.device = torch.device(device)
        self.transweather.to(self.device)
        return self

    def _preprocess_transweather(self, frame: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        TransWeather 预处理
        
        1. 保存原始尺寸
        2. 调整到32的倍数
        3. BGR -> RGB
        4. 归一化到 [-1, 1]
        5. 转换为 (B, C, H, W) 格式
        """
        original_size = frame.shape[:2]  # (H, W)
        
        # 调整到32的倍数
        h, w = original_size
        new_h = ((h + 31) // 32) * 32
        new_w = ((w + 31) // 32) * 32
        
        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # BGR -> RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [-1, 1]
        normalized = rgb.astype(np.float32) / 255.0
        normalized = (normalized - 0.5) / 0.5
        
        # 转换为 torch tensor (B, C, H, W)
        tensor = torch.from_numpy(normalized).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        
        return tensor, original_size

    def _postprocess_transweather(self, output: torch.Tensor, original_size: Tuple[int, int]) -> np.ndarray:
        """
        TransWeather 后处理
        
        1. 从 [-1, 1] 反归一化到 [0, 255]
        2. (B, C, H, W) -> (H, W, C)
        3. RGB -> BGR
        4. Resize 回原始尺寸
        """
        # 移除 batch 维度并转到CPU
        output = output.squeeze(0).cpu().detach()
        
        # (C, H, W) -> (H, W, C)
        output = output.permute(1, 2, 0).numpy()
        
        # 从 [-1, 1] 反归一化到 [0, 1]
        output = (output * 0.5) + 0.5
        output = np.clip(output, 0, 1)
        
        # 转换为 uint8
        output = (output * 255).astype(np.uint8)
        
        # RGB -> BGR
        bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        
        # Resize 回原始尺寸
        h, w = original_size
        restored = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return restored

    @torch.no_grad()
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧图像进行去雪增强
        
        处理流程:
        1. TransWeather 去雪 (PyTorch)
        
        Args:
            frame: 输入帧，BGR 格式，uint8 类型，形状为 (H, W, 3)
            
        Returns:
            增强后的帧，BGR 格式，uint8 类型
        """
        # TransWeather 去雪
        input_tensor, original_size = self._preprocess_transweather(frame)
        desnow_output = self.transweather(input_tensor)
        final_frame = self._postprocess_transweather(desnow_output, original_size)
        
        return final_frame
    
    def __repr__(self) -> str:
        return f"HDCWNetEnhancer(device={self.device}, pipeline=TransWeather)"
