import os
import uuid
from pathlib import Path
from typing import Generator, Tuple

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = BASE_DIR / "videos"
PROCESSED_DIR = BASE_DIR / "videos_processed"
POSTPROCESSED_DIR = BASE_DIR / "videos_postprocessing"

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
POSTPROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def save_upload_to_disk(filename: str, data: bytes) -> Tuple[str, Path]:
    """
    将上传的视频字节保存到本地临时目录，并返回 video_id 与路径。
    如果视频编码不兼容浏览器，则自动转码为 H.264/MP4 格式。
    """
    # 确保目录存在
    if not VIDEO_DIR.exists():
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        
    video_id = str(uuid.uuid4())
    ext = os.path.splitext(filename)[1] or ".mp4"
    temp_path = VIDEO_DIR / f"{video_id}_temp{ext}"
    final_path = VIDEO_DIR / f"{video_id}.mp4"
    
    try:
        # 先保存原始文件
        with open(temp_path, "wb") as f:
            f.write(data)
        
        # 检查视频是否需要转码
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            # 无法打开，尝试转码
            print(f"[WARN] Cannot open video {temp_path}, attempting transcode...")
            _transcode_video(temp_path, final_path)
            temp_path.unlink()  # 删除临时文件
            return video_id, final_path
        
        # 获取视频编码信息
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        cap.release()
        
        print(f"[INFO] Video codec: {codec}")
        
        # 如果不是 H.264 (avc1, h264) 或编码异常，则转码
        if codec.lower() not in ['avc1', 'h264', 'x264']:
            print(f"[INFO] Transcoding video from {codec} to H.264...")
            _transcode_video(temp_path, final_path)
            temp_path.unlink()  # 删除临时文件
            return video_id, final_path
        else:
            # 编码兼容，直接重命名
            temp_path.rename(final_path)
            return video_id, final_path
            
    except Exception as e:
        print(f"写入或转码文件失败: {temp_path}, Error: {e}")
        # 清理临时文件
        if temp_path.exists():
            temp_path.unlink()
        raise e


def _transcode_video(input_path: Path, output_path: Path) -> None:
    """
    将视频转码为浏览器兼容的 H.264/MP4 格式
    """
    try:
        import imageio
        
        # 读取原始视频
        reader = imageio.get_reader(str(input_path))
        fps = reader.get_meta_data().get('fps', 25.0)
        
        # 创建 H.264 编码的输出
        writer = imageio.get_writer(
            str(output_path),
            fps=fps,
            codec='libx264',
            pixelformat='yuv420p',
            quality=8  # 高质量
        )
        
        # 逐帧转码
        for frame in reader:
            writer.append_data(frame)
        
        writer.close()
        reader.close()
        print(f"[INFO] Transcoding completed: {output_path}")
        
    except ImportError:
        # 回退到 OpenCV 方法
        print("[WARN] imageio not available, using OpenCV for transcoding")
        cap = cv2.VideoCapture(str(input_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        cap.release()
        out.release()
        print(f"[INFO] Transcoding completed (OpenCV): {output_path}")


def get_video_path(video_id: str) -> Path:
    """
    根据 video_id 查找原始视频文件路径
    支持各种视频格式扩展名
    """
    # 尝试查找匹配的文件
    matches = list(VIDEO_DIR.glob(f"{video_id}.*"))
    if matches:
        # 返回第一个匹配的文件
        return matches[0]
    
    # 如果没有找到，打印调试信息
    print(f"[ERROR] Video not found: {video_id}")
    print(f"[DEBUG] VIDEO_DIR: {VIDEO_DIR}")
    print(f"[DEBUG] VIDEO_DIR exists: {VIDEO_DIR.exists()}")
    if VIDEO_DIR.exists():
        all_files = list(VIDEO_DIR.glob("*"))
        print(f"[DEBUG] All files in VIDEO_DIR: {[f.name for f in all_files[:10]]}")
    
    raise FileNotFoundError(f"video {video_id} not found in {VIDEO_DIR}")


def get_processed_video_path(video_id: str, post_process: bool = False) -> Path:
    base_dir = POSTPROCESSED_DIR if post_process else PROCESSED_DIR
    pattern = f"{video_id}_postprocessed.*" if post_process else f"{video_id}.*"
    for p in base_dir.glob(pattern):
        return p
    suffix = "_postprocessed" if post_process else ""
    return base_dir / f"{video_id}{suffix}.mp4"


def iter_frames(path: Path) -> Generator[Tuple[bool, np.ndarray], None, None]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        yield False, None  # type: ignore[misc]
        return
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield True, frame
    cap.release()


def count_frames(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        # fallback to manual counting if metadata not available
        frame_count = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frame_count += 1
    cap.release()
    return frame_count


def write_video(
    path: Path,
    frames: Generator[np.ndarray, None, None],
    fps: float = 25.0,
) -> None:
    first_frame = None
    for frame in frames:
        first_frame = frame
        break
    if first_frame is None:
        return

    h, w = first_frame.shape[:2]
    # 使用 imageio-ffmpeg 写入视频，支持 H.264 编码
    # 如果不可用则回退到 OpenCV
    try:
        import imageio
        import imageio.plugins.ffmpeg as ffmpeg
        
        # 创建 writer
        writer = imageio.get_writer(str(path), fps=fps, codec='libx264', pixelformat='yuv420p')
        
        # imageio 期望 RGB 格式，而 OpenCV 使用 BGR 格式，需要转换
        writer.append_data(cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB))
        for frame in frames:
            writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        writer.close()
        return
    except ImportError:
        pass
    
    # 回退到 OpenCV
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    
    writer.write(first_frame)
    for frame in frames:
        writer.write(frame)

    writer.release()

