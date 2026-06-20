"""
参数管理模块
提供动态参数配置和管理功能
"""
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ModelParameters(BaseModel):
    """模型参数配置"""

    class AODNetParams(BaseModel):
        """AOD-Net 参数"""
        use_onnx: bool = Field(default=True, description="是否使用 ONNX 推理")

    class PreNetParams(BaseModel):
        """PreNet 参数"""
        recurrent_iter: int = Field(default=6, ge=1, le=20, description="递归迭代次数")
        use_onnx: bool = Field(default=True, description="是否使用 ONNX 推理")

    class HDCWNetParams(BaseModel):
        """HDCWNet 参数"""
        use_gpu: bool = Field(default=False, description="是否使用 CUDA")

    aodnet: AODNetParams = AODNetParams()
    prenet: PreNetParams = PreNetParams()
    hdcwnet: HDCWNetParams = HDCWNetParams()


class PostProcessingParameters(BaseModel):
    """后处理参数配置"""

    class CLAHEParams(BaseModel):
        """CLAHE 参数"""
        enabled: bool = Field(default=True, description="是否启用 CLAHE")
        clip_limit: float = Field(default=1.5, ge=0.1, le=10.0, description="对比度限制")
        tile_grid_size: tuple = Field(default=(8, 8), description="网格大小")

    class UnsharpMaskParams(BaseModel):
        """非锐化掩膜参数"""
        enabled: bool = Field(default=True, description="是否启用非锐化掩膜")
        kernel_size: tuple = Field(default=(5, 5), description="核大小")
        sigma: float = Field(default=1.0, ge=0.1, le=10.0, description="高斯模糊标准差")
        amount: float = Field(default=0.8, ge=0.1, le=5.0, description="锐化强度")
        threshold: int = Field(default=5, ge=0, le=255, description="阈值")

    class DenoiseParams(BaseModel):
        """降噪参数"""
        enabled: bool = Field(default=True, description="是否启用降噪")
        diameter: int = Field(default=5, ge=1, le=15, description="滤波直径")
        sigma_color: float = Field(default=50.0, ge=1.0, le=200.0, description="颜色空间标准差")
        sigma_space: float = Field(default=50.0, ge=1.0, le=200.0, description="坐标空间标准差")

    class DynamicRangeParams(BaseModel):
        """动态范围压缩参数"""
        enabled: bool = Field(default=True, description="是否启用动态范围压缩")
        bright_threshold: int = Field(default=230, ge=0, le=255, description="亮度阈值")
        slope: float = Field(default=0.6, ge=0.1, le=1.0, description="压缩斜率")

    clahe: CLAHEParams = CLAHEParams()
    unsharp_mask: UnsharpMaskParams = UnsharpMaskParams()
    denoise: DenoiseParams = DenoiseParams()
    dynamic_range: DynamicRangeParams = DynamicRangeParams()


class DispatcherParameters(BaseModel):
    """调度器参数配置"""

    fps_threshold: float = Field(default=5.0, ge=0.1, le=60.0, description="FPS 阈值")
    frame_skip_interval: int = Field(default=30, ge=1, le=300, description="天气分类跳帧间隔")
    ema_alpha: float = Field(default=0.1, ge=0.01, le=1.0, description="EMA 平滑系数")
    enable_multi_label: bool = Field(default=False, description="是否启用多标签识别")
    multi_label_threshold: float = Field(default=0.3, ge=0.1, le=0.5, description="多标签概率阈值")


class ParameterConfig(BaseModel):
    """完整参数配置"""
    models: ModelParameters = ModelParameters()
    postprocessing: PostProcessingParameters = PostProcessingParameters()
    dispatcher: DispatcherParameters = DispatcherParameters()


class ParameterManager:
    """参数管理器"""

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # 默认配置文件路径
            backend_dir = Path(__file__).resolve().parents[1]
            self.config_path = backend_dir / "config" / "parameters.json"
        else:
            self.config_path = Path(config_path)

        # 确保配置目录存在
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # 加载或创建默认配置
        self.config = self._load_config()

    def _load_config(self) -> ParameterConfig:
        """加载配置文件"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return ParameterConfig(**data)
            except Exception as e:
                print(f"Warning: Failed to load config from {self.config_path}: {e}")
                print("Using default configuration")

        # 返回默认配置
        return ParameterConfig()

    def save_config(self) -> None:
        """保存配置到文件"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.config.dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error: Failed to save config to {self.config_path}: {e}")

    def get_config(self) -> ParameterConfig:
        """获取当前配置"""
        return self.config

    def update_config(self, updates: Dict[str, Any]) -> ParameterConfig:
        """更新配置"""
        # 将更新应用到当前配置
        config_dict = self.config.dict()
        self._deep_update(config_dict, updates)
        self.config = ParameterConfig(**config_dict)
        self.save_config()
        return self.config

    def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """深度更新字典"""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def reset_to_defaults(self) -> ParameterConfig:
        """重置为默认配置"""
        self.config = ParameterConfig()
        self.save_config()
        return self.config


# 全局参数管理器实例（延迟初始化）
_parameter_manager_instance = None


def get_parameter_manager() -> ParameterManager:
    """获取全局参数管理器实例（单例模式）"""
    global _parameter_manager_instance
    if _parameter_manager_instance is None:
        _parameter_manager_instance = ParameterManager()
    return _parameter_manager_instance


def reload_parameter_manager() -> ParameterManager:
    """重新加载参数管理器配置（用于配置更新后强制刷新）"""
    global _parameter_manager_instance
    if _parameter_manager_instance is not None:
        _parameter_manager_instance = ParameterManager()
    return get_parameter_manager()