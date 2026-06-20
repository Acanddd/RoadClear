"""
测试多标签天气识别功能

演示如何使用多标签识别来处理混合天气场景（如雨天+雾天）
"""

import cv2
import numpy as np
from pathlib import Path

from .weather_classifier import WeatherCNNClassifier
from .dispatcher import WeatherDispatcher


def test_multi_label_classification():
    """测试多标签分类功能"""
    print("=" * 80)
    print("多标签天气识别测试")
    print("=" * 80)
    
    # 初始化分类器
    classifier = WeatherCNNClassifier(device="cuda")
    
    # 测试图片路径
    test_images = [
        r"D:\acanhome\RoadClear\DAWN_dataset\rain_storm_001.jpg",
        r"D:\acanhome\RoadClear\DAWN_dataset\foggy_001.jpg",
        r"D:\acanhome\RoadClear\DAWN_dataset\snow_storm_001.jpg",
    ]
    
    print("\n测试不同阈值下的多标签识别：\n")
    
    for img_path in test_images:
        if not Path(img_path).exists():
            print(f"[跳过] 图片不存在: {img_path}")
            continue
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"[错误] 无法读取图片: {img_path}")
            continue
        
        # 预测
        pred = classifier.predict(img)
        
        print(f"图片: {Path(img_path).name}")
        print(f"  单标签结果: {pred.label} ({pred.probs[['fog', 'rain', 'snow'].index(pred.label)]*100:.1f}%)")
        print(f"  完整概率: Fog={pred.probs[0]*100:.1f}%, Rain={pred.probs[1]*100:.1f}%, Snow={pred.probs[2]*100:.1f}%")
        
        # 测试不同阈值
        for threshold in [0.2, 0.3, 0.4]:
            multi_labels = pred.get_multi_labels(threshold)
            if len(multi_labels) > 1:
                desc = pred.get_weather_description(threshold)
                print(f"  阈值={threshold}: 混合天气 - {desc}")
            else:
                print(f"  阈值={threshold}: 单一天气 - {pred.weather_str}")
        
        print()


def test_multi_label_dispatcher():
    """测试调度器的多标签处理"""
    print("=" * 80)
    print("多标签调度器测试")
    print("=" * 80)
    
    # 初始化调度器
    dispatcher = WeatherDispatcher()
    
    # 创建测试图片（模拟雨天+雾天场景）
    test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n测试场景 1: 启用多标签识别")
    enhanced, weather, model_chain = dispatcher.process_frame(
        test_img,
        video_id="test_multi",
        enable_multi_label=True,
        multi_label_threshold=0.3,
    )
    print(f"  天气: {weather}")
    print(f"  模型链: {model_chain}")
    
    print("\n测试场景 2: 禁用多标签识别（传统单标签模式）")
    enhanced, weather, model_chain = dispatcher.process_frame(
        test_img,
        video_id="test_single",
        enable_multi_label=False,
    )
    print(f"  天气: {weather}")
    print(f"  模型链: {model_chain}")
    
    print("\n测试场景 3: 调整阈值（更宽松，更容易触发多标签）")
    enhanced, weather, model_chain = dispatcher.process_frame(
        test_img,
        video_id="test_loose",
        enable_multi_label=True,
        multi_label_threshold=0.2,
    )
    print(f"  天气: {weather}")
    print(f"  模型链: {model_chain}")


def analyze_dawn_multi_label():
    """分析 DAWN 数据集中的潜在混合天气样本"""
    print("=" * 80)
    print("DAWN 数据集混合天气分析")
    print("=" * 80)
    
    classifier = WeatherCNNClassifier(device="cuda")
    dataset_path = Path(r"D:\acanhome\RoadClear\DAWN_dataset")
    
    if not dataset_path.exists():
        print(f"数据集路径不存在: {dataset_path}")
        return
    
    # 收集所有图片
    image_files = list(dataset_path.rglob("*.jpg")) + list(dataset_path.rglob("*.png"))
    
    # 统计混合天气样本
    mixed_weather_samples = []
    threshold = 0.3
    
    print(f"\n扫描 {len(image_files)} 张图片，寻找混合天气样本（阈值={threshold}）...\n")
    
    for img_path in image_files[:100]:  # 只测试前100张
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        pred = classifier.predict(img)
        multi_labels = pred.get_multi_labels(threshold)
        
        if len(multi_labels) > 1:
            mixed_weather_samples.append({
                "path": img_path.name,
                "labels": multi_labels,
                "description": pred.get_weather_description(threshold),
            })
    
    print(f"发现 {len(mixed_weather_samples)} 个混合天气样本：\n")
    
    for i, sample in enumerate(mixed_weather_samples[:20], 1):  # 显示前20个
        print(f"{i}. {sample['path']}")
        print(f"   {sample['description']}")
        labels_str = ", ".join([f"{label}={prob*100:.1f}%" for label, prob in sample['labels']])
        print(f"   详细: {labels_str}")
        print()
    
    if len(mixed_weather_samples) > 20:
        print(f"... 还有 {len(mixed_weather_samples) - 20} 个样本未显示")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试多标签天气识别功能")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["classify", "dispatch", "analyze", "all"],
        help="测试模式",
    )
    
    args = parser.parse_args()
    
    if args.mode in ["classify", "all"]:
        test_multi_label_classification()
        print("\n")
    
    if args.mode in ["dispatch", "all"]:
        test_multi_label_dispatcher()
        print("\n")
    
    if args.mode in ["analyze", "all"]:
        analyze_dawn_multi_label()


if __name__ == "__main__":
    main()
