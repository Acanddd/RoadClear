import os
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_model import BaseEnhanceModel

# PReNet 预训练权重路径
# 默认指向仓库根目录下的 PReNet-master（如果你只用 ONNX 推理，这个权重不会被加载）。
# 注意：不要依赖当前工作目录 (cwd)。改为基于源码文件位置推导 backend/仓库根目录，避免路径漂移。
_BACKEND_DIR = Path(__file__).resolve().parents[2]  # .../backend
_REPO_ROOT = _BACKEND_DIR.parent
PRETRAINED_WEIGHTS_PATH = str(
    _REPO_ROOT / "PReNet-master" / "logs" / "Rain100H" / "PReNet6" / "net_latest.pth"
)

# ONNX 模型默认路径（位于 backend/onnx_models）
ONNX_MODEL_PATH = str(_BACKEND_DIR / "onnx_models" / "prenet.onnx")


class PReNet(nn.Module):
    """
    完整的 PReNet 网络结构
    来自: https://github.com/cdwren/PReNet
    """

    def __init__(self, recurrent_iter=6, use_GPU=True):
        super(PReNet, self).__init__()
        self.iteration = recurrent_iter
        self.use_GPU = use_GPU

        self.conv0 = nn.Sequential(
            nn.Conv2d(6, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.res_conv5 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU()
        )
        self.conv_i = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_f = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv_g = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Tanh()
        )
        self.conv_o = nn.Sequential(
            nn.Conv2d(32 + 32, 32, 3, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(32, 3, 3, 1, 1),
        )

    def forward(self, input):
        batch_size, row, col = input.size(0), input.size(2), input.size(3)

        x = input
        h = torch.zeros(batch_size, 32, row, col)
        c = torch.zeros(batch_size, 32, row, col)

        if self.use_GPU:
            h = h.to(input.device)
            c = c.to(input.device)

        x_list = []
        for i in range(self.iteration):
            x = torch.cat((input, x), 1)
            x = self.conv0(x)

            x = torch.cat((x, h), 1)
            i = self.conv_i(x)
            f = self.conv_f(x)
            g = self.conv_g(x)
            o = self.conv_o(x)
            c = f * c + i * g
            h = o * torch.tanh(c)

            x = h
            resx = x
            x = F.relu(self.res_conv1(x) + resx)
            resx = x
            x = F.relu(self.res_conv2(x) + resx)
            resx = x
            x = F.relu(self.res_conv3(x) + resx)
            resx = x
            x = F.relu(self.res_conv4(x) + resx)
            resx = x
            x = F.relu(self.res_conv5(x) + resx)
            x = self.conv(x)

            x = x + input
            x_list.append(x)

        return x, x_list


class PReNetONNXWrapper:
    """
    PReNet ONNX Runtime 推理封装类
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
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 获取模型信息
        self.input_shape = self.session.get_inputs()[0].shape

        print(f"[PReNetONNXWrapper] Model loaded from: {model_path}")
        print(f"[PReNetONNXWrapper] Providers: {providers}")
        print(f"[PReNetONNXWrapper] Input shape: {self.input_shape}")
        print(f"[PReNetONNXWrapper] Output names: {self.output_names}")

    def run(self, input_data: np.ndarray) -> np.ndarray:
        """
        运行推理

        Args:
            input_data: 输入数组，形状为 (B, C, H, W)，float32 类型

        Returns:
            输出数组，形状为 (B, C, H, W)
        """
        # ONNX 输出顺序: [output, intermediate_outputs]
        # 我们只需要最后一个 output
        outputs = self.session.run(self.output_names, {self.input_name: input_data})
        # 第一个输出是最终的 output
        return outputs[0]


class PreNetEnhancer(BaseEnhanceModel):
    """
    PReNet 去雨模型封装

    支持两种推理模式:
    1. PyTorch 模式 (use_onnx=False): 使用 PyTorch 模型进行推理
    2. ONNX 模式 (use_onnx=True): 使用 ONNX Runtime 进行推理，速度更快

    使用在 Rain100H 数据集上预训练的 PReNet6 模型权重，
    提供高质量的图像去雨功能。
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        weights_path: str | None = None,
        recurrent_iter: int = 6,
        use_onnx: bool = False,
        onnx_path: str | None = None
    ) -> None:
        self.device = torch.device(device)
        self.use_onnx = use_onnx
        self._onnx_wrapper: Optional[PReNetONNXWrapper] = None
        self.recurrent_iter = recurrent_iter

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
                providers = ['CPUExecutionProvider']
                if str(device).startswith("cuda"):
                    try:
                        import onnxruntime as ort
                        if 'CUDAExecutionProvider' in ort.get_available_providers():
                            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                    except ImportError:
                        pass

                self._onnx_wrapper = PReNetONNXWrapper(onnx_path, providers)
                print(f"[PreNetEnhancer] Using ONNX Runtime for inference")
            except Exception as e:
                raise RuntimeError(f"ONNX load failed: {e}") from e

        # PyTorch 模式 (或 fallback)
        if not use_onnx:
            # 判断是否使用 GPU
            use_gpu = str(device).startswith("cuda") or str(device) == "cuda"

            # 创建 PReNet 模型
            self.model = PReNet(recurrent_iter=recurrent_iter, use_GPU=use_gpu)

            # 加载预训练权重
            if os.path.exists(weights_path):
                try:
                    state_dict = torch.load(weights_path, map_location=device, weights_only=False)
                    # 处理可能的 DataParallel 包装
                    if all(k.startswith('module.') for k in state_dict.keys()):
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    self.model.load_state_dict(state_dict)
                    print(f"[PreNetEnhancer] Loaded pretrained weights from: {weights_path}")
                except Exception as e:
                    raise RuntimeError(f"Weight load failed: {e}") from e
            else:
                raise FileNotFoundError(weights_path)

            self.model.to(self.device)
            self.model.eval()

    def to(self, device: Any) -> "PreNetEnhancer":
        self.device = torch.device(device)

        if not self.use_onnx:
            self.model.to(self.device)

            # 更新模型的 use_GPU 标志
            if str(device).startswith("cuda"):
                self.model.use_GPU = True
            else:
                self.model.use_GPU = False

        return self

    @torch.no_grad()
    def enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        对单帧图像进行去雨增强

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
            out = torch.clamp(out, 0., 1.)
        else:
            # PyTorch 推理模式
            tensor = tensor.to(self.device)

            # PReNet 推理
            out, _ = self.model(tensor)
            out = torch.clamp(out, 0., 1.)

            # 转换为 numpy
            out = out.cpu()

        # 转换为 numpy 并反归一化
        if not torch.isfinite(out).all():
            raise RuntimeError("Non-finite model output")
        out = out.squeeze(0).permute(1, 2, 0).numpy()
        out = np.clip(out * 255.0, 0, 255).astype(np.uint8)

        # RGB -> BGR
        bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        return bgr

    @property
    def is_onnx_mode(self) -> bool:
        """返回是否使用 ONNX 模式"""
        return self.use_onnx

    def __repr__(self) -> str:
        mode = "ONNX" if self.use_onnx else "PyTorch"
        return f"PreNetEnhancer(mode={mode}, device={self.device}, recurrent_iter={self.recurrent_iter})"
