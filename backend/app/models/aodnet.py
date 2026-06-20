import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from .base_model import BaseEnhanceModel

# AOD-Net PyTorch 预训练权重路径 / ONNX 模型路径
# 注意：不要依赖当前工作目录 (cwd)。改为基于源码文件位置推导 backend 目录，避免在 uvicorn/任务调度等场景下路径漂移。
_BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
PRETRAINED_WEIGHTS_PATH = str(_BACKEND_DIR / "pretrained_weights" / "aodnet_state_dict.pth")
ONNX_MODEL_PATH = str(_BACKEND_DIR / "onnx_models" / "aodnet.onnx")


class AODNet(nn.Module):
    """
    完整的 AOD-Net 网络结构 (PyTorch 版本)
    来自: https://github.com/0lnetworkuser/AOD-Net_pytorch
    """
    
    def __init__(self):
        super(AODNet, self).__init__()

        self.relu = nn.ReLU(inplace=True)
    
        self.conv1 = nn.Conv2d(3, 3, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(3, 3, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(6, 3, 5, 1, 2, bias=True)
        self.conv4 = nn.Conv2d(6, 3, 7, 1, 3, bias=True)
        self.conv5 = nn.Conv2d(12, 3, 3, 1, 1, bias=True)
        
    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x1))

        concat1 = torch.cat((x1, x2), 1)
        x3 = self.relu(self.conv3(concat1))

        concat2 = torch.cat((x2, x3), 1)
        x4 = self.relu(self.conv4(concat2))

        concat3 = torch.cat((x1, x2, x3, x4), 1)
        x5 = self.relu(self.conv5(concat3))

        clean_image = self.relu((x5 * x) - x5 + 1) 
        
        return clean_image


class AODNetONNXWrapper:
    """
    AOD-Net ONNX Runtime 推理封装类
    提供更快的推理速度
    """
    
    def __init__(
        self,
        model_path: str,
        providers: Optional[list] = None
    ):
        """
        初始化 ONNX Runtime 会话
        
        Args:
            model_path: ONNX 模型文件路径
            providers: 执行提供者列表，默认自动选择
        """
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime not installed. Run: pip install onnxruntime")
        
        # 设置会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4
        
        # 自动选择providers
        if providers is None:
            providers = ['CPUExecutionProvider']
            # 尝试添加 CUDA 支持
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # 创建推理会话
        self.session = ort.InferenceSession(model_path, sess_options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # 获取模型信息
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_shape = self.session.get_outputs()[0].shape
        
        print(f"[AODNetONNXWrapper] Model loaded from: {model_path}")
        print(f"[AODNetONNXWrapper] Providers: {providers}")
        print(f"[AODNetONNXWrapper] Input shape: {self.input_shape}")
    
    def run(self, input_data: np.ndarray) -> np.ndarray:
        """
        运行推理
        
        Args:
            input_data: 输入数组，形状为 (B, C, H, W)，float32 类型
            
        Returns:
            输出数组，形状为 (B, C, H, W)
        """
        return self.session.run([self.output_name], {self.input_name: input_data})[0]


class AODNetEnhancer(BaseEnhanceModel):
    """
    AOD-Net 去雾模型封装
    
    支持两种推理模式:
    1. PyTorch 模式 (use_onnx=False): 使用 PyTorch 模型进行推理
    2. ONNX 模式 (use_onnx=True): 使用 ONNX Runtime 进行推理，速度更快
    
    使用 PyTorch 版本 AOD-Net 预训练模型权重，
    提供高质量的图像去雾功能。
    """

    def __init__(
        self, 
        device: str | torch.device = "cpu",
        weights_path: str | None = None,
        use_onnx: bool = False,
        onnx_path: str | None = None
    ) -> None:
        self.device = torch.device(device)
        self.use_onnx = use_onnx
        self._onnx_wrapper: Optional[AODNetONNXWrapper] = None
        
        # 默认使用预训练权重路径
        if weights_path is None:
            weights_path = PRETRAINED_WEIGHTS_PATH
        
        # ONNX 模式
        if use_onnx:
            if onnx_path is None:
                onnx_path = ONNX_MODEL_PATH
            
            if not os.path.exists(onnx_path):
                raise FileNotFoundError(f"ONNX model not found at: {onnx_path}")
            
            try:
                # 设置 ONNX providers
                providers = None
                if str(device).startswith("cuda"):
                    try:
                        import onnxruntime as ort
                        if 'CUDAExecutionProvider' in ort.get_available_providers():
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    except ImportError:
                        pass
                
                self._onnx_wrapper = AODNetONNXWrapper(onnx_path, providers)
                print(f"[AODNetEnhancer] Using ONNX Runtime for inference")
            except Exception as e:
                print(f"[AODNetEnhancer] Warning: Failed to load ONNX model: {e}")
                print("[AODNetEnhancer] Falling back to PyTorch mode")
                self.use_onnx = False
                use_onnx = False
        
        # PyTorch 模式 (或 fallback)
        if not use_onnx:
            # 创建 AOD-Net 模型
            self.model = AODNet()
            
            # 加载预训练权重
            if os.path.exists(weights_path):
                try:
                    # 加载完整 checkpoint (可能包含 epoch 信息或整个模型)
                    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
                    
                    # 检查加载的内容类型
                    if isinstance(checkpoint, AODNet):
                        # 权重文件直接保存的是整个模型
                        self.model = checkpoint
                        print(f"[AODNetEnhancer] Loaded pretrained model from: {weights_path}")
                    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                        # 是完整 checkpoint 包含 state_dict
                        self.model.load_state_dict(checkpoint['state_dict'])
                        print(f"[AODNetEnhancer] Loaded state_dict from: {weights_path}")
                    elif isinstance(checkpoint, dict):
                        # 尝试找到匹配的键
                        self.model.load_state_dict(checkpoint)
                        print(f"[AODNetEnhancer] Loaded weights from dict: {weights_path}")
                    else:
                        # 直接是模型 state_dict
                        self.model = checkpoint
                        print(f"[AODNetEnhancer] Loaded model directly from: {weights_path}")
                        
                except Exception as e:
                    print(f"[AODNetEnhancer] Warning: Failed to load weights from {weights_path}: {e}")
                    print("[AODNetEnhancer] Using randomly initialized weights.")
            else:
                print(f"[AODNetEnhancer] Warning: Weights file not found at {weights_path}")
                print("[AODNetEnhancer] Using randomly initialized weights.")
            
            self.model.to(self.device)
            self.model.eval()

    def to(self, device: Any) -> "AODNetEnhancer":
        """切换设备"""
        if not self.use_onnx:
            self.device = torch.device(device)
            self.model.to(self.device)
        # ONNX 模式下设备由 ONNX Runtime 管理
        return self

    @torch.no_grad()
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧图像进行去雾增强
        
        Args:
            frame: 输入帧，BGR 格式，uint8 类型，形状为 (H, W, 3)
            
        Returns:
            增强后的帧，BGR 格式，uint8 类型
        """
        # BGR -> RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1]，并转换为 tensor
        rgb = rgb.astype(np.float32) / 255.0
        tensor = torch.from_numpy(rgb).float()
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        
        if self.use_onnx:
            # ONNX 推理模式
            input_data = tensor.numpy()
            
            # 推理
            out = self._onnx_wrapper.run(input_data)
            out = torch.from_numpy(out)
        else:
            # PyTorch 推理模式
            tensor = tensor.to(self.device)
            
            # AOD-Net 推理
            out = self.model(tensor)
            out = torch.clamp(out, 0., 1.)
            
            # 转换为 numpy
            out = out.cpu()
        
        # 转换为 numpy 并反归一化
        out = out.squeeze(0).permute(1, 2, 0).numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
        
        # RGB -> BGR
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        
        # 后处理：去噪
        # 使用双边滤波保留边缘的同时去除噪声
        bgr = cv2.bilateralFilter(bgr, d=2, sigmaColor=30, sigmaSpace=30)
        
        return bgr
    
    @property
    def is_onnx_mode(self) -> bool:
        """返回是否使用 ONNX 模式"""
        return self.use_onnx
    
    def __repr__(self) -> str:
        mode = "ONNX" if self.use_onnx else "PyTorch"
        return f"AODNetEnhancer(mode={mode}, device={self.device})"
