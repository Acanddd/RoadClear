"""
测试天气分类器在 DAWN 数据集上的性能

DAWN 数据集标签映射规则：
- dusttornado, foggy, haze, mist → fog
- rain_storm → rain
- snow_storm → snow
- sand_storm → 跳过
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    classification_report,
)
from tqdm import tqdm

from .weather_classifier import WeatherCNNClassifier, WeatherType


# DAWN 数据集标签映射
DAWN_LABEL_MAPPING = {
    "dusttornado": "fog",
    "foggy": "fog",
    "haze": "fog",
    "mist": "fog",
    "rain_storm": "rain",
    "snow_storm": "snow",
    "sand_storm": None,  # 跳过
}


def extract_label_from_filename(filename: str) -> str | None:
    """
    从 DAWN 数据集的文件名中提取天气标签。
    
    示例文件名格式：
    - foggy_001.jpg
    - rain_storm_042.png
    - dusttornado_123.jpg
    
    :param filename: 文件名（不含路径）
    :return: 映射后的标签 (fog/rain/snow) 或 None（跳过）
    """
    filename_lower = filename.lower()
    
    # 按照最长匹配优先
    for dawn_label, mapped_label in DAWN_LABEL_MAPPING.items():
        if dawn_label in filename_lower:
            return mapped_label
    
    return None


def load_dawn_dataset(dataset_path: str | Path) -> List[Tuple[Path, str]]:
    """
    加载 DAWN 数据集，返回 (图片路径, 标签) 列表。
    
    :param dataset_path: DAWN 数据集根目录
    :return: [(图片路径, 标签), ...]
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"数据集路径不存在: {dataset_path}")
    
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    samples = []
    
    # 递归查找所有图片
    for img_path in dataset_path.rglob("*"):
        if img_path.suffix.lower() not in image_extensions:
            continue
        
        label = extract_label_from_filename(img_path.name)
        if label is None:
            continue  # 跳过 sand_storm 或无法识别的文件
        
        samples.append((img_path, label))
    
    return samples


def evaluate_classifier(
    classifier: WeatherCNNClassifier,
    dataset_path: str | Path,
    output_dir: str | Path = None,
    enable_multi_label: bool = True,
    multi_label_threshold: float = 0.3,
) -> Dict:
    """
    在 DAWN 数据集上评估天气分类器。
    
    :param classifier: 天气分类器实例
    :param dataset_path: DAWN 数据集路径
    :param output_dir: 结果保存目录（可选）
    :param enable_multi_label: 是否启用多标签评估
    :param multi_label_threshold: 多标签阈值
    :return: 评估结果字典
    """
    print(f"[测试] 加载 DAWN 数据集: {dataset_path}")
    samples = load_dawn_dataset(dataset_path)
    
    if not samples:
        raise ValueError(f"未找到有效样本，请检查数据集路径: {dataset_path}")
    
    print(f"[测试] 找到 {len(samples)} 个有效样本")
    print(f"[测试] 多标签模式: {'启用' if enable_multi_label else '禁用'}, 阈值: {multi_label_threshold}")
    
    # 统计各类别样本数
    label_counts = {"fog": 0, "rain": 0, "snow": 0}
    for _, label in samples:
        label_counts[label] += 1
    print(f"[测试] 样本分布: {label_counts}")
    
    # 预测
    y_true = []
    y_pred = []
    y_pred_multi = []  # 多标签预测结果
    failed_samples = []
    
    print("[测试] 开始预测...")
    for img_path, true_label in tqdm(samples, desc="预测进度"):
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[警告] 无法读取图片: {img_path}")
                failed_samples.append((str(img_path), "读取失败"))
                continue
            
            prediction = classifier.predict(img)
            pred_label = prediction.label
            
            y_true.append(true_label)
            y_pred.append(pred_label)
            
            # 多标签预测
            if enable_multi_label:
                multi_labels = prediction.get_multi_labels(multi_label_threshold)
                y_pred_multi.append([label for label, prob in multi_labels])
            else:
                y_pred_multi.append([pred_label])
            
        except Exception as e:
            print(f"[错误] 处理图片失败 {img_path}: {e}")
            failed_samples.append((str(img_path), str(e)))
    
    if not y_true:
        raise ValueError("没有成功预测的样本")
    
    print(f"[测试] 成功预测 {len(y_true)} 个样本，失败 {len(failed_samples)} 个")
    
    # 计算指标
    labels = ["fog", "rain", "snow"]
    
    # 1. 单标签指标（传统方式）
    overall_accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="weighted", zero_division=0
    )
    
    # 2. 多标签指标
    multi_label_metrics = {}
    if enable_multi_label:
        # 部分匹配准确率：预测的多标签中包含真实标签
        partial_match = sum(1 for true, pred_list in zip(y_true, y_pred_multi) if true in pred_list)
        partial_match_accuracy = partial_match / len(y_true)
        
        # 完全匹配准确率：预测只有一个标签且正确
        exact_match = sum(1 for true, pred_list in zip(y_true, y_pred_multi) 
                         if len(pred_list) == 1 and pred_list[0] == true)
        exact_match_accuracy = exact_match / len(y_true)
        
        # 混合天气识别率：识别出多个天气的比例
        multi_weather_rate = sum(1 for pred_list in y_pred_multi if len(pred_list) > 1) / len(y_pred_multi)
        
        # 各类别的多标签召回率
        multi_label_recall = {}
        for label in labels:
            true_count = sum(1 for t in y_true if t == label)
            if true_count > 0:
                recalled = sum(1 for t, pred_list in zip(y_true, y_pred_multi) 
                              if t == label and label in pred_list)
                multi_label_recall[label] = recalled / true_count
            else:
                multi_label_recall[label] = 0.0
        
        multi_label_metrics = {
            "partial_match_accuracy": float(partial_match_accuracy),
            "exact_match_accuracy": float(exact_match_accuracy),
            "multi_weather_rate": float(multi_weather_rate),
            "multi_label_recall": {k: float(v) for k, v in multi_label_recall.items()},
        }
    
    # 组织结果
    results = {
        "overall_accuracy": float(overall_accuracy),
        "total_samples": len(y_true),
        "failed_samples": len(failed_samples),
        "label_distribution": label_counts,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            labels[i]: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(labels))
        },
        "macro_avg": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1_score": float(f1_macro),
        },
        "weighted_avg": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1_score": float(f1_weighted),
        },
        "multi_label_metrics": multi_label_metrics,
    }
    
    # 打印结果
    print("\n" + "=" * 80)
    print("天气分类器性能评估结果 (DAWN 数据集)")
    print("=" * 80)
    print(f"\n【单标签模式】")
    print(f"总体准确率 (Overall Accuracy): {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    print(f"测试样本数: {len(y_true)}")
    print(f"失败样本数: {len(failed_samples)}")
    
    if enable_multi_label and multi_label_metrics:
        print(f"\n【多标签模式】(阈值={multi_label_threshold})")
        print(f"部分匹配准确率: {multi_label_metrics['partial_match_accuracy']:.4f} ({multi_label_metrics['partial_match_accuracy']*100:.2f}%)")
        print(f"  → 预测的多标签中包含真实标签即算正确")
        print(f"完全匹配准确率: {multi_label_metrics['exact_match_accuracy']:.4f} ({multi_label_metrics['exact_match_accuracy']*100:.2f}%)")
        print(f"  → 预测只有一个标签且完全正确")
        print(f"混合天气识别率: {multi_label_metrics['multi_weather_rate']:.4f} ({multi_label_metrics['multi_weather_rate']*100:.2f}%)")
        print(f"  → 识别出多个天气的样本比例")
        print(f"\n多标签召回率 (真实标签被识别出的比例):")
        for label in labels:
            recall_val = multi_label_metrics['multi_label_recall'][label]
            print(f"  {label:>6}: {recall_val:.4f} ({recall_val*100:.2f}%)")
    
    print("\n混淆矩阵 (Confusion Matrix):")
    print(f"{'':>10} {'fog':>10} {'rain':>10} {'snow':>10}")
    for i, label in enumerate(labels):
        row = cm[i]
        print(f"{label:>10} {row[0]:>10} {row[1]:>10} {row[2]:>10}")
    
    print("\n各类别指标 (Per-Class Metrics):")
    print(f"{'类别':>10} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-" * 66)
    for label in labels:
        metrics = results["per_class_metrics"][label]
        print(
            f"{label:>10} "
            f"{metrics['precision']:>12.4f} "
            f"{metrics['recall']:>12.4f} "
            f"{metrics['f1_score']:>12.4f} "
            f"{metrics['support']:>10}"
        )
    
    print("-" * 66)
    print(
        f"{'宏平均':>10} "
        f"{results['macro_avg']['precision']:>12.4f} "
        f"{results['macro_avg']['recall']:>12.4f} "
        f"{results['macro_avg']['f1_score']:>12.4f} "
        f"{len(y_true):>10}"
    )
    print(
        f"{'加权平均':>10} "
        f"{results['weighted_avg']['precision']:>12.4f} "
        f"{results['weighted_avg']['recall']:>12.4f} "
        f"{results['weighted_avg']['f1_score']:>12.4f} "
        f"{len(y_true):>10}"
    )
    
    # 使用 sklearn 的分类报告（更详细）
    print("\n详细分类报告 (Classification Report):")
    print(classification_report(y_true, y_pred, labels=labels, target_names=labels, digits=4))
    
    # 保存结果
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存 JSON 结果
        json_path = output_dir / "evaluation_results.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n[保存] 结果已保存到: {json_path}")
        
        # 保存失败样本列表
        if failed_samples:
            failed_path = output_dir / "failed_samples.txt"
            with open(failed_path, "w", encoding="utf-8") as f:
                for img_path, error in failed_samples:
                    f.write(f"{img_path}\t{error}\n")
            print(f"[保存] 失败样本列表: {failed_path}")
        
        # 保存混淆矩阵为 CSV
        cm_path = output_dir / "confusion_matrix.csv"
        np.savetxt(
            cm_path,
            cm,
            delimiter=",",
            header=",".join(labels),
            comments="",
            fmt="%d",
        )
        print(f"[保存] 混淆矩阵: {cm_path}")
    
    print("=" * 80)
    
    return results


def main():
    """主函数：运行评估"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试天气分类器在 DAWN 数据集上的性能")
    parser.add_argument(
        "--dataset",
        type=str,
        default=r"D:\acanhome\RoadClear\DAWN_dataset",
        help="DAWN 数据集路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="模型权重路径（可选，默认使用预设路径）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="运行设备",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./evaluation_results",
        help="结果保存目录",
    )
    parser.add_argument(
        "--enable-multi-label",
        action="store_true",
        default=True,
        help="启用多标签评估",
    )
    parser.add_argument(
        "--disable-multi-label",
        action="store_true",
        help="禁用多标签评估（使用传统单标签模式）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="多标签阈值 (0.1-0.5)",
    )
    
    args = parser.parse_args()
    
    # 处理多标签开关
    enable_multi_label = args.enable_multi_label and not args.disable_multi_label
    
    # 初始化分类器
    print(f"[初始化] 加载天气分类器 (device={args.device})")
    classifier = WeatherCNNClassifier(
        device=args.device,
        model_path=args.model,
    )
    
    # 运行评估
    evaluate_classifier(
        classifier=classifier,
        dataset_path=args.dataset,
        output_dir=args.output,
        enable_multi_label=enable_multi_label,
        multi_label_threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
