import base64
import io
import time
from typing import Any, Dict, List

import numpy as np
from fastapi import APIRouter, HTTPException

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    matplotlib.rcParams['axes.unicode_minus'] = False
except Exception:
    plt = None

from ..evaluation.detector import DetectionResult, detector, get_license_plate_detector
from ..evaluation.metrics import (
    compute_contrast_gain,
    compute_detection_count_change_rate,
    compute_high_confidence_count,
    compute_mean_confidence,
    compute_noise_suppression_rate,
    compute_temporal_stability,
)
from ..scheduler.dispatcher import dispatcher
from ..utils.video_utils import get_processed_video_path, get_video_path, iter_frames

router = APIRouter()


def _build_html_report(
    video_id: str,
    metrics_before: Dict[str, Any],
    metrics_after: Dict[str, Any],
    image_metrics: Dict[str, float],
    base_fps: float,
    enh_fps: float,
) -> str:
    """生成基于无 GT 真实指标的评估报告。"""

    chart_html = ""
    if plt is not None:
        labels = ["平均置信度", "时序稳定性"]
        before_vals = [metrics_before["mean_confidence"], metrics_before["temporal_stability"]]
        after_vals = [metrics_after["mean_confidence"], metrics_after["temporal_stability"]]

        x = np.arange(len(labels))
        width = 0.3

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        ax1 = axes[0]
        ax1.bar(x - width / 2, before_vals, width, label='增强前', color='#e74c3c')
        ax1.bar(x + width / 2, after_vals, width, label='增强后', color='#2ecc71')
        ax1.set_ylabel('得分')
        ax1.set_title('检测质量对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels)
        ax1.legend()
        # 动态设置Y轴范围，聚焦在数据范围附近
        all_vals = before_vals + after_vals
        min_val = min(all_vals)
        max_val = max(all_vals)
        margin = (max_val - min_val) * 0.3 if max_val > min_val else 0.1
        ax1.set_ylim(max(0, min_val - margin), min(1.0, max_val + margin))
        # 设置更细的Y轴刻度，间隔0.01
        y_min = max(0, min_val - margin)
        y_max = min(1.0, max_val + margin)
        ax1.set_yticks(np.arange(np.floor(y_min * 100) / 100, np.ceil(y_max * 100) / 100 + 0.01, 0.01))
        ax1.grid(axis='y', linestyle='--', alpha=0.3)

        ax2 = axes[1]
        img_labels = ["对比度增益", "噪声抑制率"]
        img_vals = [image_metrics["contrast_gain"] * 100, image_metrics["noise_suppression_rate"] * 100]
        colors = ['#3498db' if v >= 0 else '#e67e22' for v in img_vals]
        ax2.bar(img_labels, img_vals, color=colors)
        ax2.set_ylabel('百分比 (%)')
        ax2.set_title('图像质量改善')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='--', alpha=0.3)

        ax3 = axes[2]
        hc_labels = ["增强前", "增强后"]
        hc_vals = [metrics_before["high_conf_count"], metrics_after["high_conf_count"]]
        ax3.bar(hc_labels, hc_vals, color=['#e74c3c', '#2ecc71'])
        ax3.set_ylabel('检测数量')
        ax3.set_title('高置信度检测数对比 (≥0.7)')
        # 动态设置Y轴范围，聚焦在数据范围附近
        min_hc = min(hc_vals) if hc_vals else 0
        max_hc = max(hc_vals) if hc_vals else 100
        margin = (max_hc - min_hc) * 0.3 if max_hc > min_hc else max_hc * 0.2
        ax3.set_ylim(max(0, min_hc - margin), max_hc + margin)
        # 设置合适的刻度间隔
        y_range = max_hc + margin - max(0, min_hc - margin)
        if y_range > 200:
            step = 50
        elif y_range > 100:
            step = 20
        elif y_range > 50:
            step = 10
        else:
            step = 5
        ax3.set_yticks(np.arange(max(0, int(min_hc - margin)), int(max_hc + margin) + step, step))
        ax3.grid(axis='y', linestyle='--', alpha=0.3)

        buf = io.BytesIO()
        fig.tight_layout()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")

        chart_html = (
            '<div class="chart-container">'
            '<h3>可视化分析</h3>'
            f'<img src="data:image/png;base64,{img_b64}" alt="metrics_chart"'
            ' style="max-width:100%;border-radius:8px;box-shadow:0 4px 8px rgba(0,0,0,0.1);" />'
            '</div>'
        )

    det_change_pct = image_metrics["detection_count_change_rate"] * 100
    det_change_sign = "+" if det_change_pct >= 0 else ""

    conf_diff = metrics_after["mean_confidence"] - metrics_before["mean_confidence"]
    hc_diff = metrics_after["high_conf_count"] - metrics_before["high_conf_count"]
    stab_diff = metrics_after["temporal_stability"] - metrics_before["temporal_stability"]
    lp_diff = metrics_after["license_plate_count"] - metrics_before["license_plate_count"]

    html = _render_report_html(
        video_id, metrics_before, metrics_after, image_metrics,
        chart_html, det_change_sign, det_change_pct,
        conf_diff, hc_diff, stab_diff, lp_diff, base_fps, enh_fps,
    )
    return html


def _render_report_html(
    video_id: str,
    m_before: Dict[str, Any],
    m_after: Dict[str, Any],
    img_m: Dict[str, float],
    chart_html: str,
    det_sign: str,
    det_pct: float,
    conf_diff: float,
    hc_diff: int,
    stab_diff: float,
    lp_diff: int,
    base_fps: float,
    enh_fps: float,
) -> str:
    """拼装 HTML 报告字符串。"""
    css = (
        '<style>'
        '.roadclear-report{font-family:"Helvetica Neue",Helvetica,Arial,sans-serif;line-height:1.6;color:#333;width:100%;box-sizing:border-box}'
        '.roadclear-report .card{background:#fff;border-radius:12px;padding:20px;border:1px solid #ebeef5;margin-bottom:20px}'
        '.roadclear-report h2,.roadclear-report h3{color:#2c3e50;margin-top:0}'
        '.roadclear-report .metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:15px;margin:15px 0}'
        '.roadclear-report .metric-item{background:#f0f4f8;padding:12px;border-radius:8px;text-align:center;border-left:4px solid #3498db}'
        '.roadclear-report .metric-value{font-size:20px;font-weight:bold;color:#2980b9;display:block}'
        '.roadclear-report .metric-label{font-size:12px;color:#7f8c8d}'
        '.roadclear-report table{border-collapse:collapse;width:100%;margin-top:12px;background:#fff}'
        '.roadclear-report th,.roadclear-report td{border:1px solid #eee;padding:8px;font-size:12px;text-align:center;color:#000}'
        '.roadclear-report th{background-color:#34495e;color:#fff}'
        '.roadclear-report tr:nth-child(even){background-color:#fafafa}'
        '.roadclear-report .badge{padding:3px 6px;border-radius:4px;font-size:11px;font-weight:bold}'
        '.roadclear-report .badge-up{background:#e8f5e9;color:#2e7d32}'
        '.roadclear-report .badge-down{background:#fce4ec;color:#c62828}'
        '.roadclear-report .info-box{background:#e3f2fd;border-left:4px solid #1976d2;padding:10px;font-size:12px;margin:15px 0;color:#000}'
        '.roadclear-report .chart-container{text-align:center;margin-bottom:20px}'
        '</style>'
    )

    def _badge(val, is_int=False):
        cls = "badge-up" if val >= 0 else "badge-down"
        sign = "+" if val >= 0 else ""
        if is_int:
            return f'<span class="badge {cls}">{sign}{val}</span>'
        return f'<span class="badge {cls}">{sign}{val:.4f}</span>'

    def _badge_pct(val):
        cls = "badge-up" if val >= 0 else "badge-down"
        sign = "+" if val >= 0 else ""
        return f'<span class="badge {cls}">{sign}{val:.2f}%</span>'

    ts = time.strftime('%Y-%m-%d %H:%M:%S')
    parts = [
        f'<div class="roadclear-report">{css}',
        '<div class="card">',
        '<h2>RoadClear 无标注评估报告</h2>',
        f'<p style="font-size:12px;color:#666"><b>视频 ID</b>: {video_id} | <b>生成时间</b>: {ts}</p>',
        '<div class="info-box">本报告所有指标均基于真实计算得出，无需 Ground Truth 标注数据。'
        '评估维度覆盖检测置信度、检测数量、图像质量和时序稳定性。</div>',
        '</div>',

        '<div class="card"><h3>核心指标概览</h3><div class="metric-grid">',
        f'<div class="metric-item"><span class="metric-value">{m_before["mean_confidence"]:.3f} → {m_after["mean_confidence"]:.3f}</span>'
        '<span class="metric-label">平均置信度</span></div>',
        f'<div class="metric-item"><span class="metric-value">{m_before["high_conf_count"]} → {m_after["high_conf_count"]}</span>'
        '<span class="metric-label">高置信度检测数 (≥0.7)</span></div>',
        f'<div class="metric-item"><span class="metric-value">{det_sign}{det_pct:.1f}%</span>'
        '<span class="metric-label">检测数量变化率</span></div>',
        f'<div class="metric-item"><span class="metric-value">{img_m["contrast_gain"]*100:.1f}%</span>'
        '<span class="metric-label">对比度增益</span></div>',
        f'<div class="metric-item"><span class="metric-value">{img_m["noise_suppression_rate"]*100:.1f}%</span>'
        '<span class="metric-label">噪声抑制率</span></div>',
        f'<div class="metric-item"><span class="metric-value">{m_after["temporal_stability"]:.3f}</span>'
        '<span class="metric-label">时序稳定性</span></div>',
        '</div></div>',

        chart_html,

        '<div class="card"><h3>详细指标对比</h3><table>',
        '<thead><tr><th>指标</th><th>增强前</th><th>增强后</th><th>变化</th></tr></thead><tbody>',
        f'<tr><td>平均置信度</td><td>{m_before["mean_confidence"]:.4f}</td>'
        f'<td>{m_after["mean_confidence"]:.4f}</td><td>{_badge(conf_diff)}</td></tr>',
        f'<tr><td>高置信度检测数 (≥0.7)</td><td>{m_before["high_conf_count"]}</td>'
        f'<td>{m_after["high_conf_count"]}</td><td>{_badge(hc_diff, is_int=True)}</td></tr>',
        f'<tr><td>总检测数</td><td>{m_before["total_detections"]}</td>'
        f'<td>{m_after["total_detections"]}</td><td>{_badge_pct(det_pct)}</td></tr>',
        f'<tr><td>时序稳定性</td><td>{m_before["temporal_stability"]:.4f}</td>'
        f'<td>{m_after["temporal_stability"]:.4f}</td><td>{_badge(stab_diff)}</td></tr>',
        '<tr><td>对比度增益</td><td colspan="2" style="text-align:center">增强后相对增强前</td>'
        f'<td>{_badge_pct(img_m["contrast_gain"]*100)}</td></tr>',
        '<tr><td>噪声抑制率</td><td colspan="2" style="text-align:center">增强后相对增强前</td>'
        f'<td>{_badge_pct(img_m["noise_suppression_rate"]*100)}</td></tr>',
        '</tbody></table></div>',

        '<div class="card"><h3>性能信息</h3><table>',
        '<thead><tr><th>项目</th><th>值</th></tr></thead><tbody>',
        f'<tr><td>增强前处理 FPS</td><td>{base_fps:.2f}</td></tr>',
        f'<tr><td>增强后处理 FPS</td><td>{enh_fps:.2f}</td></tr>',
        '</tbody></table>',
        '<div class="info-box" style="margin-top:10px">',
        '<b>说明</b>：运行 FPS 指的是评估系统处理视频的速度（包含检测、增强等操作），',
        '而非原始视频的帧率。评估时需要逐帧进行目标检测和图像分析，',
        '因此处理速度会低于原始帧率。FPS 越高说明评估处理越快。',
        '</div>',
        '</div>',

        '</div>',
    ]
    return "\n".join(parts)


def _evaluate_video(video_id: str) -> Dict[str, object]:
    """
    无 Ground Truth 评估：所有指标均由真实检测与图像分析计算得出。
    核心指标：平均置信度、高置信度检测数、检测数量变化率、车牌识别数量、
            图像对比度增益、噪声抑制率、时序稳定性。
    """
    try:
        src_path = get_video_path(video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="原始视频不存在")

    processed_path = get_processed_video_path(video_id)
    if not processed_path.exists():
        processed_path = get_processed_video_path(video_id, post_process=True)
    has_processed = processed_path.exists()

    if not has_processed:
        raise HTTPException(409, 'Process the video successfully before evaluation')
    from ..model_registry import require
    try:
        require('vehicle')
        lp_detector = require('plate')
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc

    base_dets_list: List[List[DetectionResult]] = []
    base_lp_dets_list: List[List[DetectionResult]] = []
    base_frames: List[np.ndarray] = []
    start_base = time.time()
    for ok, frame in iter_frames(src_path):
        if not ok:
            break
        base_frames.append(frame.copy())
        base_dets_list.append(detector.detect(frame))
        if lp_detector:
            base_lp_dets_list.append(lp_detector.detect(frame, conf_thres=0.3))
    end_base = time.time()

    enhanced_dets_list: List[List[DetectionResult]] = []
    enhanced_lp_dets_list: List[List[DetectionResult]] = []
    enhanced_frames: List[np.ndarray] = []
    start_enh = time.time()
    if has_processed:
        for ok, frame in iter_frames(processed_path):
            if not ok:
                break
            enhanced_frames.append(frame.copy())
            enhanced_dets_list.append(detector.detect(frame))
            if lp_detector:
                enhanced_lp_dets_list.append(lp_detector.detect(frame, conf_thres=0.3))
    else:
        for ok, frame in iter_frames(src_path):
            if not ok:
                break
            enhanced, _, _ = dispatcher.process_frame(frame, video_id=f"eval-{video_id}")
            enhanced_frames.append(enhanced.copy())
            enhanced_dets_list.append(detector.detect(enhanced))
            if lp_detector:
                enhanced_lp_dets_list.append(lp_detector.detect(enhanced, conf_thres=0.3))
    end_enh = time.time()

    if not base_frames or len(base_frames) != len(enhanced_frames):
        raise HTTPException(422, "Video frames are empty or misaligned")
    min_len = min(len(base_dets_list), len(enhanced_dets_list))
    base_dets_aligned = base_dets_list[:min_len]
    enh_dets_aligned = enhanced_dets_list[:min_len]
    base_frames_aligned = base_frames[:min_len]
    enh_frames_aligned = enhanced_frames[:min_len]
    base_lp_dets_aligned = base_lp_dets_list[:min_len] if lp_detector else []
    enh_lp_dets_aligned = enhanced_lp_dets_list[:min_len] if lp_detector else []

    # 计算车牌数量
    base_lp_count = sum(len(f) for f in base_lp_dets_aligned) if lp_detector else 0
    enh_lp_count = sum(len(f) for f in enh_lp_dets_aligned) if lp_detector else 0

    metrics_before = {
        "mean_confidence": compute_mean_confidence(base_dets_aligned),
        "high_conf_count": compute_high_confidence_count(base_dets_aligned),
        "temporal_stability": compute_temporal_stability(base_dets_aligned),
        "total_detections": sum(len(f) for f in base_dets_aligned),
        "license_plate_count": base_lp_count,
    }

    metrics_after = {
        "mean_confidence": compute_mean_confidence(enh_dets_aligned),
        "high_conf_count": compute_high_confidence_count(enh_dets_aligned),
        "temporal_stability": compute_temporal_stability(enh_dets_aligned),
        "total_detections": sum(len(f) for f in enh_dets_aligned),
        "license_plate_count": enh_lp_count,
    }

    image_metrics = {
        "detection_count_change_rate": compute_detection_count_change_rate(
            base_dets_aligned, enh_dets_aligned
        ),
        "contrast_gain": compute_contrast_gain(base_frames_aligned, enh_frames_aligned),
        "noise_suppression_rate": compute_noise_suppression_rate(
            base_frames_aligned, enh_frames_aligned
        ),
    }

    base_fps = min_len / (end_base - start_base) if end_base > start_base else 0.0
    enh_fps = min_len / (end_enh - start_enh) if end_enh > start_enh else 0.0

    html_report = _build_html_report(
        video_id=video_id,
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        image_metrics=image_metrics,
        base_fps=base_fps,
        enh_fps=enh_fps,
    )

    return {
        "video_id": video_id,
        "metrics": {
            "before": metrics_before,
            "after": metrics_after,
            "image_quality": image_metrics,
            "performance": {"base_fps": base_fps, "enhanced_fps": enh_fps},
        },
        "html_report": html_report,
    }


@router.post('/evaluate/{video_id}')
def evaluate_video(video_id: str):
    from ..operation import operation
    with operation():
        return _evaluate_video(video_id)
