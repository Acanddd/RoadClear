from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

# 预设的权重文件路径，可通过 training 脚本生成
DEFAULT_WEATHER_CLASSIFIER_WEIGHTS = Path(__file__).resolve().parent / "trained_models_rscm" / "weather_classifier_mobilenet_v3_small.pth"

# 与训练脚本一致：类别索引 0=fog, 1=rain, 2=snow（RSCM 中 rainy 文件夹映射到索引 1）
CLASS_INDEX_TO_TAG = ("fog", "rain", "snow")

WeatherType = Literal["fog", "rain", "snow"]


class WeatherClassifierModel(nn.Module):
    """
    使用 MobileNet V2 / V3 作为骨干网络进行三分类（单标签：fog / rain / snow）。
    直接继承自 torchvision 模型。
    """

    def __init__(
        self,
        backbone: str = "mobilenet_v3_small",
        num_labels: int = 3,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        backbone = backbone.lower()
        if backbone == "mobilenet_v3_small":
            self.backbone = torchvision.models.mobilenet_v3_small(pretrained=pretrained)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_labels)
        elif backbone == "mobilenet_v3_large":
            self.backbone = torchvision.models.mobilenet_v3_large(pretrained=pretrained)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_labels)
        elif backbone == "mobilenet_v2":
            self.backbone = torchvision.models.mobilenet_v2(pretrained=pretrained)
            in_features = self.backbone.classifier[-1].in_features
            self.backbone.classifier[-1] = nn.Linear(in_features, num_labels)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    def forward(self, x):
        return self.backbone(x)


@dataclass
class WeatherPrediction:
    label: WeatherType
    probs: Tuple[float, float, float]  # (P(fog), P(rain), P(snow)) softmax
    weather_str: str  # 可读字符串，如 "Fog"
    multi_labels: list[tuple[WeatherType, float]] = None  # 多标签结果：[(label, prob), ...]，按概率降序排列

    def get_multi_labels(self, threshold: float = 0.3) -> list[tuple[WeatherType, float]]:
        """
        获取多标签结果（概率超过阈值的所有类别）

        :param threshold: 概率阈值，默认 0.3 (30%)
        :return: [(label, prob), ...] 按概率降序排列
        """
        if self.multi_labels is not None:
            return self.multi_labels

        labels_with_probs = [
            (CLASS_INDEX_TO_TAG[i], self.probs[i])  # type: ignore[misc]
            for i in range(len(self.probs))
            if self.probs[i] >= threshold
        ]
        # 按概率降序排列
        labels_with_probs.sort(key=lambda x: x[1], reverse=True)
        return labels_with_probs

    def is_mixed_weather(self, threshold: float = 0.3) -> bool:
        """判断是否为混合天气（多个类别概率都超过阈值）"""
        return len(self.get_multi_labels(threshold)) > 1

    def get_weather_description(self, threshold: float = 0.3) -> str:
        """
        获取天气描述字符串

        单标签: "Rain"
        多标签: "Rain(60%) + Fog(40%)"
        """
        multi = self.get_multi_labels(threshold)
        if len(multi) <= 1:
            return self.weather_str

        parts = [f"{label.capitalize()}({prob*100:.0f}%)" for label, prob in multi]
        return " + ".join(parts)

    def get_processing_labels(self, threshold: float = 0.3) -> list[tuple[WeatherType, float]]:
        """
        获取实际用于处理的天气标签列表

        规则：
        1. 如果雾天（fog）概率最大，只返回雾天标签，以保证画面稳定
        2. 否则返回所有超过阈值的标签（用于复合天气联合处理）

        :param threshold: 概率阈值，默认 0.3 (30%)
        :return: [(label, prob), ...] 按概率降序排列
        """
        multi = self.get_multi_labels(threshold)

        if len(multi) == 0:
            return []

        top_label, top_prob = multi[0]
        if top_label == "fog":
            return [("fog", top_prob)]

        return multi

    def should_use_single_model(self, threshold: float = 0.3) -> bool:
        """
        判断是否应该使用单一模型处理

        :return: True 表示使用单一模型，False 表示可以使用多模型联合处理
        """
        multi = self.get_multi_labels(threshold)
        if len(multi) <= 1:
            return True

        top_label, _ = multi[0]
        return top_label == "fog"


class WeatherCNNClassifier:
    """
    基于 MobileNet 的天气分类器：单标签三分类 fog / rain / snow（与 CrossEntropy 训练一致）。
    """

    def __init__(
        self,
        device: str | torch.device = "cpu",
        model_path: Optional[str] = None,
        backbone: str = "mobilenet_v3_small",
        pretrained_backbone: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.model = WeatherClassifierModel(
            backbone=backbone,
            num_labels=3,
            pretrained=pretrained_backbone,
        ).to(self.device)

        if model_path is None:
            model_path = str(DEFAULT_WEATHER_CLASSIFIER_WEIGHTS)

        if not model_path or not Path(model_path).is_file():
            raise FileNotFoundError(f"Weather weights missing: {model_path}")
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            # 处理键名兼容性：训练时保存的可能是直接的torchvision模型state_dict
            if not any(key.startswith("backbone.") for key in state_dict.keys()):
                new_state_dict = {}
                for key, value in state_dict.items():
                    new_key = f"backbone.{key}"
                    new_state_dict[new_key] = value
                state_dict = new_state_dict
            self.model.load_state_dict(state_dict)

        self.model.eval()
        self.input_size = 224
        self.transform = transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    @torch.no_grad()
    def predict(self, image: np.ndarray) -> WeatherPrediction:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs_t = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        idx = int(np.argmax(probs_t))

        label: WeatherType = CLASS_INDEX_TO_TAG[idx]  # type: ignore[assignment]
        fog_prob = float(probs_t[0])
        rain_prob = float(probs_t[1])
        snow_prob = float(probs_t[2])

        weather_str = label.capitalize()

        return WeatherPrediction(
            label=label,
            probs=(fog_prob, rain_prob, snow_prob),
            weather_str=weather_str,
        )

    @torch.no_grad()
    def predict_multi(self, image: np.ndarray) -> WeatherPrediction:
        """与 predict 相同；保留名称以兼容旧调用。"""
        return self.predict(image)
