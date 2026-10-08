from dataclasses import dataclass
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..utils.progress_logger import add_log
from ..models.aodnet import AODNetEnhancer
from ..models.prenet import PreNetEnhancer
from .. import model_registry
from ..config_manager import get_parameter_manager
from .weather_classifier import WeatherCNNClassifier, WeatherPrediction, WeatherType


# 自动检测最佳设备
def get_device() -> str:
    """自动选择最佳设备 (CUDA > CPU)"""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        print(f"[WeatherDispatcher] Using GPU: {device_name}")
        return "cuda"
    print("[WeatherDispatcher] Using CPU")
    return "cpu"


# 全局设备配置
from ..settings import DEVICE


@dataclass
class ModelStats:
    name: str
    ema_fps: float = 0.0
    last_elapsed: float = 0.0
    call_count: int = 0


class ModelPool:
    """
    简单模型池管理：支持动态添加/移除增强模型。
    """

    def __init__(self) -> None:
        self._models: Dict[str, object] = {}

    def add_model(self, name: str, model: object) -> None:
        self._models[name] = model

    def remove_model(self, name: str) -> None:
        if name in self._models:
            del self._models[name]

    def get(self, name: str) -> Optional[object]:
        return self._models.get(name)

    def list_models(self) -> Dict[str, str]:
        return {name: type(m).__name__ for name, m in self._models.items()}


class WeatherDispatcher:
    """

    - WeatherCNNClassifier 单标签输出 fog / rain / snow；
    - 每种天气对应单一增强模型（雾 AODNet、雨 PreNet、雪 TransWeather）；
    - 支持每 N 帧分类一次，降低天气判定开销；
    - 统计各模型 FPS 供观测，不向 simple 自动降级；
    - 模型缺失时拒绝处理；用户主动选择 simple 时走轻量增强逻辑。
    """

    def __init__(self, fps_threshold: float = None, device: str = None) -> None:
        # 获取配置参数
        config = get_parameter_manager().get_config()

        # 使用配置参数或默认值
        self.fps_threshold = fps_threshold if fps_threshold is not None else config.dispatcher.fps_threshold
        self.frame_skip_interval = config.dispatcher.frame_skip_interval
        self.ema_alpha = config.dispatcher.ema_alpha

        # 自动选择设备
        self.device = device if device else DEVICE

        model_registry.initialize()
        self.classifier = model_registry.instances.get('weather')
        self.pool = ModelPool()
        for name in ('aodnet', 'prenet', 'transweather'):
            if name in model_registry.instances:
                self.pool.add_model(name, model_registry.instances[name])

        # 统计信息
        self.model_stats: Dict[str, ModelStats] = {
            "aodnet": ModelStats(name="aodnet"),
            "prenet": ModelStats(name="prenet"),
            "transweather": ModelStats(name="transweather"),
            "simple": ModelStats(name="simple"),
        }
        self._video_states: Dict[str, Dict[str, object]] = {}

    def _update_fps(self, model_name: str, elapsed_sec: float) -> float:
        stats = self.model_stats[model_name]
        stats.last_elapsed = elapsed_sec
        stats.call_count += 1
        if elapsed_sec <= 0:
            return stats.ema_fps
        inst_fps = 1.0 / elapsed_sec
        if stats.ema_fps <= 0:
            stats.ema_fps = inst_fps
        else:
            stats.ema_fps = (
                (1 - self.ema_alpha) * stats.ema_fps + self.ema_alpha * inst_fps
            )
        return stats.ema_fps

    def _get_video_state(self, video_id: Optional[str]) -> Optional[Dict[str, object]]:
        if not video_id:
            return None
        return self._video_states.setdefault(
            video_id,
            {
                "frame_count": 0,
                "last_weather": None,
                "last_weather_str": "",
                "last_model_chain": None,
            },
        )

    def _simple_enhance(self, frame: np.ndarray) -> np.ndarray:
        """
        轻量级备用增强：使用轻微锐化/保边滤波。
        """
        return cv2.bilateralFilter(frame, d=5, sigmaColor=50, sigmaSpace=50)

    def _select_models_by_weather(self, weather: str) -> List[str]:
        """单标签天气 -> 单一模型。"""
        if weather == "fog":
            return ["aodnet"]
        if weather == "rain":
            return ["prenet"]
        if weather == "snow":
            return ["transweather"]
        return ["simple"]

    def _select_models_by_multi_weather(
        self,
        multi_labels: list[tuple[str, float]],
        threshold: float = 0.3
    ) -> List[str]:
        """
        多标签天气 -> 级联模型链

        根据天气类别的概率从高到低排序，决定模型处理顺序。
        例如：[(rain, 0.6), (fog, 0.4)] -> ["prenet", "aodnet"]

        :param multi_labels: [(label, prob), ...] 按概率降序排列
        :param threshold: 概率阈值，低于此值的类别不参与处理
        :return: 模型链列表
        """
        model_chain = []
        weather_to_model = {
            "fog": "aodnet",
            "rain": "prenet",
            "snow": "transweather",
        }

        for label, prob in multi_labels:
            if prob >= threshold:
                model_name = weather_to_model.get(label)
                if model_name and model_name not in model_chain:
                    model_chain.append(model_name)

        return model_chain if model_chain else ["simple"]

    def _parse_forced_model(self, forced_model: str) -> List[str]:
        """
        解析用户指定的强制模型。
        支持的格式：aodnet, prenet, hdcwnet, simple, aodnet->prenet, prenet->aodnet, hdcwnet->prenet 等
        """
        parts = [p.strip().lower() for p in forced_model.split('->')]
        parts = ['transweather' if p == 'hdcwnet' else p for p in parts]
        if not parts or any(p not in ('aodnet', 'prenet', 'transweather', 'simple') for p in parts):
            raise ValueError('Unknown model selection')
        if 'simple' in parts and len(parts) != 1:
            raise ValueError('simple cannot be chained')
        return parts

    def require_selection(self, selection):
        if selection in (None, 'auto'):
            if not model_registry.capabilities()['auto']['available']:
                raise RuntimeError(model_registry.capabilities()['auto']['reason'])
            return
        for name in self._parse_forced_model(selection):
            if name != 'simple' and self.pool.get(name) is None:
                raise RuntimeError(f'{name} unavailable')

    def process_frame(
        self,
        frame: np.ndarray,
        video_id: Optional[str] = None,
        forced_model: Optional[str] = None,
    ) -> Tuple[np.ndarray, WeatherType | str, str]:
        """
        对输入帧进行天气识别并调用对应增强模型。

        :param frame: 单帧图像，BGR，uint8。
        :param video_id: 当前视频 ID，用于记录调度日志。
        :param forced_model: 强制使用的模型，可选值：None/"auto", "aodnet", "prenet", "transweather", "simple", "aodnet->prenet", "prenet->aodnet", "hdcwnet->prenet" 等
        :return: (增强后图像, 天气标签 fog/rain/snow 或强制模式下的 \"forced\", 实际使用的模型链名称)
        """
        self.require_selection(forced_model)
        # 从配置中读取多标签参数
        config = get_parameter_manager().get_config()
        self.frame_skip_interval = config.dispatcher.frame_skip_interval
        enable_multi_label = config.dispatcher.enable_multi_label
        multi_label_threshold = config.dispatcher.multi_label_threshold

        video_state = self._get_video_state(video_id)
        if forced_model and forced_model != "auto":
            model_chain = self._parse_forced_model(forced_model)
            weather: WeatherType | str = "forced"
            weather_str = "forced"
            is_forced = True
        else:
            should_classify = True
            if video_state is not None:
                frame_count = video_state["frame_count"]
                if frame_count % self.frame_skip_interval != 0 and video_state["last_weather"] is not None:
                    should_classify = False
            if should_classify:
                pred: WeatherPrediction = self.classifier.predict(frame)
                weather = pred.label
                weather_str = pred.weather_str

                # 多标签处理
                if enable_multi_label:
                    # 使用 get_processing_labels 应用雾天优先规则
                    processing_labels = pred.get_processing_labels(multi_label_threshold)
                    all_labels = pred.get_multi_labels(multi_label_threshold)

                    if len(processing_labels) > 1:
                        # 混合天气，使用级联模型链
                        model_chain = self._select_models_by_multi_weather(processing_labels, multi_label_threshold)
                        weather_str = pred.get_weather_description(multi_label_threshold)
                        add_log(
                            video_id,
                            f"检测到混合天气: {weather_str}, 使用级联模型链: {' -> '.join(model_chain)}",
                        )
                    elif len(processing_labels) == 1 and len(all_labels) > 1:
                        # 雾天优先规则生效：检测到混合天气但雾天权重最大，只使用去雾
                        model_chain = self._select_models_by_weather(processing_labels[0][0])
                        all_weather_str = pred.get_weather_description(multi_label_threshold)
                        weather_str = f"Fog (优先)"
                        add_log(
                            video_id,
                            f"检测到混合天气: {all_weather_str}, 但雾天权重最大，仅使用去雾处理以保证画面稳定",
                        )
                    else:
                        # 单一天气
                        model_chain = self._select_models_by_weather(weather)
                else:
                    # 单标签模式
                    model_chain = self._select_models_by_weather(weather)

                if video_state is not None:
                    video_state["last_weather"] = weather
                    video_state["last_weather_str"] = weather_str
                    video_state["last_model_chain"] = model_chain
            else:
                weather = video_state["last_weather"]  # type: ignore[assignment]
                weather_str = video_state["last_weather_str"]  # type: ignore[assignment]
                model_chain = video_state["last_model_chain"]  # type: ignore[assignment]

                # 检查缓存的模型链是否与当前配置一致
                # 如果配置已经关闭多标签，但缓存中有多模型链，则重新选择模型
                if model_chain and len(model_chain) > 1 and not enable_multi_label:
                    # 配置已关闭多标签，但缓存中有多模型链，重新根据单标签模式选择
                    if isinstance(weather, str) and weather in ("fog", "rain", "snow"):
                        model_chain = self._select_models_by_weather(weather)
                        video_state["last_model_chain"] = model_chain
                        add_log(
                            video_id,
                            f"检测到配置变化（多标签已关闭），重新选择模型: {weather} -> {model_chain[0]}",
                        )

                if not model_chain and isinstance(weather, str) and weather in ("fog", "rain", "snow"):
                    model_chain = self._select_models_by_weather(weather)
                if not model_chain:
                    model_chain = ["simple"]
                model_desc = " -> ".join(model_chain) if model_chain else "simple"
                add_log(
                    video_id,
                    f"weather={weather_str}, 重用上次分类结果, 模型链={model_desc}",
                )
            is_forced = False

        if video_state is not None:
            video_state["frame_count"] += 1

        if len(model_chain) == 1:
            model_str = model_chain[0]
        else:
            model_str = " -> ".join(model_chain)

        # 简单路径
        if model_chain == ["simple"]:
            start_t = perf_counter()
            enhanced = self._simple_enhance(frame)
            elapsed = perf_counter() - start_t
            fps = self._update_fps("simple", elapsed)
            add_log(
                video_id,
                f"weather={weather_str}, 使用简单增强 simple, 估计 FPS={fps:.1f}",
            )
            return enhanced, weather, model_str

        total_start = perf_counter()
        result = frame
        model_details = []
        step_perf = []

        for model_name in model_chain:
            model = self.pool.get(model_name)
            if model is None:
                raise RuntimeError(f'Model {model_name} unavailable')

            step_start = perf_counter()
            result = model.enhance(result)
            step_elapsed = perf_counter() - step_start
            self._update_fps(model_name, step_elapsed)
            model_details.append(model_name)
            step_perf.append(f"{model_name} FPS={1.0/step_elapsed:.1f}")

        total_elapsed = perf_counter() - total_start
        overall_fps = 1.0 / total_elapsed if total_elapsed > 0 else 0
        model_chain_str = " -> ".join(model_details)

        add_log(
            video_id,
            f"weather={weather_str}, 使用模型链 [{model_chain_str}], "
            f"总耗时={total_elapsed:.3f}s, 总FPS={overall_fps:.1f}, 步骤: {', '.join(step_perf)}",
        )

        return result, weather, model_chain_str


# 全局调度器实例，便于复用与模型池管理
dispatcher = WeatherDispatcher()
