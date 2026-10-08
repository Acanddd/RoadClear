from typing import List

from .detector import DetectionResult


def compute_lpr_accuracy(
    base_dets: List[List[DetectionResult]],
    enhanced_dets: List[List[DetectionResult]],
) -> float:
    """
    车牌识别准确率的简化/模拟实现。

    由于未集成真实车牌检测与 OCR，这里采用一个近似策略：
    - 将原始视频检测结果视作“参考”；
    - 统计增强后检测中，与参考检测 IoU 足够高且类别一致的框占参考总框数的比例；
    - 将该比例视作“车牌识别准确率”的 proxy，主要用于趋势对比，而非真实 LPR 性能。
    """
    total_ref = 0
    total_matched = 0

    from .metrics import _iou  # 复用 IoU 计算

    for gt_frame, enh_frame in zip(base_dets, enhanced_dets):
        total_ref += len(gt_frame)
        for g in gt_frame:
            best_iou = 0.0
            for p in enh_frame:
                if p.cls != g.cls:
                    continue
                iou_val = _iou(g.bbox, p.bbox)
                if iou_val > best_iou:
                    best_iou = iou_val
            if best_iou >= 0.5:
                total_matched += 1

    if total_ref == 0:
        return 1.0
    return total_matched / total_ref
