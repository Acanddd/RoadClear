from typing import Dict, Generator

import asyncio
import cv2
from fastapi import APIRouter, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from ..scheduler.dispatcher import dispatcher
from ..utils.progress_logger import add_log, get_logs
from ..utils.video_utils import (
    count_frames,
    get_processed_video_path,
    get_video_path,
    iter_frames,
    save_upload_to_disk,
    write_video,
)
from ..postprocessing import postprocess_frame

router = APIRouter()

# 简单内存索引，记录视频是否已处理
VIDEO_STATUS: Dict[str, Dict[str, bool | str]] = {}


@router.post("/upload_video")
async def upload_video(file: UploadFile) -> Dict[str, str]:
    """
    接收视频文件并保存到临时目录，返回 video_id。
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名为空")

    try:
        data = await file.read()
        video_id, path = save_upload_to_disk(file.filename, data)
        VIDEO_STATUS[video_id] = {"processed": False, "path": str(path)}
        return {"video_id": video_id}
    except Exception as e:
        import traceback
        print(f"ERROR: Upload failed: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/process/{video_id}")
async def process_video(
    video_id: str, 
    model: str = "auto", 
    post_process: bool = False,
    enable_multi_label: bool = None,
    multi_label_threshold: float = None
) -> Dict[str, str | int | bool]:
    """
    对指定视频进行逐帧增强，并写回到 processed 目录。
    返回处理帧数与下载链接占位符。

    :param video_id: 视频 ID
    :param model: 使用的模型，可选值：auto(自动), aodnet, prenet, simple, aodnet->prenet, prenet->aodnet
    :param post_process: 是否对增强结果执行后处理（CLAHE / Unsharp Mask / 降噪 / 动态范围压缩）
    :param enable_multi_label: 是否启用多标签天气识别
    :param multi_label_threshold: 多标签识别的概率阈值
    """
    # 如果提供了多标签参数，更新配置
    if enable_multi_label is not None or multi_label_threshold is not None:
        from ..config_manager import get_parameter_manager, reload_parameter_manager
        param_manager = get_parameter_manager()
        updates = {}
        if enable_multi_label is not None:
            updates['enable_multi_label'] = enable_multi_label
        if multi_label_threshold is not None:
            updates['multi_label_threshold'] = multi_label_threshold
        if updates:
            param_manager.update_config({'dispatcher': updates})
            # 重新加载配置，确保 dispatcher 使用最新配置
            reload_parameter_manager()
            add_log(video_id, f"配置已更新: enable_multi_label={enable_multi_label}, threshold={multi_label_threshold}")
    try:
        src_path = get_video_path(video_id)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="视频不存在")

    total_frames = count_frames(src_path)
    VIDEO_STATUS.setdefault(video_id, {})["total_frames"] = total_frames
    VIDEO_STATUS.setdefault(video_id, {})["status"] = "processing"
    VIDEO_STATUS.setdefault(video_id, {})["post_processed"] = post_process

    dst_path = get_processed_video_path(video_id, post_process=post_process)

    # 记录用户选择的模型
    user_model = model if model != "auto" else None

    def frame_generator():
        frame_count = 0
        model_chain_used = None
        for ok, frame in iter_frames(src_path):
            if not ok:
                break
            try:
                enhanced, _weather, _model = dispatcher.process_frame(
                    frame, video_id=video_id, forced_model=user_model
                )
                if post_process:
                    enhanced = postprocess_frame(enhanced)
                
                # 记录第一个非 simple 的模型链
                if model_chain_used is None and _model != "simple":
                    model_chain_used = _model
            except Exception as e:
                # 记录异常并回退为原始帧以避免整个任务失败
                add_log(video_id, f"帧处理异常，使用原始帧回退: {e}")
                enhanced = frame
            
            frame_count += 1
            VIDEO_STATUS.setdefault(video_id, {})["frames_processed"] = frame_count
            yield enhanced
        # 存储模型链信息
        if model_chain_used:
            VIDEO_STATUS[video_id]["model_chain"] = model_chain_used

    try:
        write_video(dst_path, frame_generator())
    except Exception as e:
        VIDEO_STATUS.setdefault(video_id, {})["status"] = "failed"
        add_log(video_id, f"视频处理失败: {e}")
        raise HTTPException(status_code=500, detail=f"视频处理失败: {e}")

    VIDEO_STATUS.setdefault(video_id, {})["processed"] = True
    VIDEO_STATUS[video_id]["processed_path"] = str(dst_path)
    VIDEO_STATUS[video_id]["status"] = "completed"

    # 返回可通过静态文件服务访问的URL
    video_filename = dst_path.name
    if post_process:
        download_url = f"/postprocessed/{video_filename}"
    else:
        download_url = f"/processed/{video_filename}"

    # 获取模型链信息
    model_chain = VIDEO_STATUS[video_id].get("model_chain", "simple")

    return {
        "video_id": video_id,
        "frames_processed": VIDEO_STATUS[video_id].get("frames_processed", 0),
        "total_frames": total_frames,
        "download_url": download_url,
        "model_chain": model_chain,
        "post_process": post_process,
        "status": VIDEO_STATUS[video_id].get("status", "completed"),
    }


def _stream_enhanced_frames(video_id: str, model: str = "auto", post_process: bool = False) -> Generator[bytes, None, None]:
    """
    以 multipart/x-mixed-replace 形式流式返回增强后视频（逐帧 JPEG）。
    """
    try:
        processed_path = get_processed_video_path(video_id, post_process=post_process)
        if processed_path.exists():
            cap_path = processed_path
        else:
            cap_path = get_video_path(video_id)
    except FileNotFoundError:
        return

    user_model = model if model != "auto" else None

    for ok, frame in iter_frames(cap_path):
        if not ok:
            break
        # 若是原始视频或后处理文件不存在，则实时增强并可选后处理
        if not processed_path.exists():
            try:
                frame, _weather, _model = dispatcher.process_frame(
                    frame, video_id=video_id, forced_model=user_model
                )
                if post_process:
                    frame = postprocess_frame(frame)
            except Exception as e:
                add_log(video_id, f"实时流帧处理异常，使用原始帧回退: {e}")
                # keep original frame

        ret, buf = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        jpg_bytes = buf.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpg_bytes + b"\r\n"
        )


@router.get("/stream/{video_id}")
async def stream_video(video_id: str, model: str = "auto", post_process: bool = False):
    """
    以 MJPEG 流形式返回增强后视频。
    """
    generator = _stream_enhanced_frames(video_id, model=model, post_process=post_process)
    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/status/{video_id}")
async def get_video_status(video_id: str) -> Dict[str, object]:
    """
    返回指定视频处理状态和进度信息。
    """
    status = VIDEO_STATUS.get(video_id, {})
    return {
        "video_id": video_id,
        "status": status.get("status", "unknown"),
        "frames_processed": status.get("frames_processed", 0),
        "total_frames": status.get("total_frames", 0),
        "post_processed": status.get("post_processed", False),
        "processed": status.get("processed", False),
        "model_chain": status.get("model_chain", "simple"),
    }


@router.get("/logs/{video_id}")
async def get_video_logs(video_id: str) -> Dict[str, object]:
    """
    返回指定视频的调度/进度日志列表，用于前端轮询展示。
    """
    logs = get_logs(video_id)
    return {"video_id": video_id, "logs": logs}


@router.get("/original/{video_id}")
async def get_original_video(video_id: str):
    """
    获取原始视频文件，支持 Range 请求以便浏览器流式播放
    """
    from fastapi.responses import FileResponse
    import os
    
    try:
        print(f"[DEBUG] Requesting original video: {video_id}")
        
        # 尝试获取视频路径
        try:
            original_path = get_video_path(video_id)
            print(f"[DEBUG] Found video at: {original_path}")
        except FileNotFoundError:
            print(f"[ERROR] Video file not found for video_id: {video_id}")
            # 列出 videos 目录中的文件以便调试
            from pathlib import Path
            videos_dir = Path(__file__).resolve().parents[1] / "videos"
            print(f"[DEBUG] Videos directory: {videos_dir}")
            if videos_dir.exists():
                files = list(videos_dir.glob(f"{video_id}.*"))
                print(f"[DEBUG] Files matching pattern '{video_id}.*': {files}")
            raise HTTPException(status_code=404, detail=f"原始视频不存在，video_id: {video_id}")
        
        if not original_path.exists():
            print(f"[ERROR] Video file path exists in glob but file not found: {original_path}")
            raise HTTPException(status_code=404, detail="原始视频文件不存在")
        
        # 根据文件扩展名确定 media_type
        ext = original_path.suffix.lower()
        media_type_map = {
            '.mp4': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
        }
        media_type = media_type_map.get(ext, 'video/mp4')
        
        print(f"[DEBUG] Serving video with media_type: {media_type}, size: {original_path.stat().st_size} bytes")
        
        # 使用 FileResponse，它会自动处理 Range 请求
        return FileResponse(
            path=str(original_path),
            media_type=media_type,
            filename=f"original_{video_id}{ext}",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        print(f"[ERROR] Unexpected error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.get("/download/{video_id}")
async def download_video(video_id: str):
    """
    下载增强后的视频文件，支持 Range 请求以便浏览器流式播放
    """
    from fastapi.responses import FileResponse
    
    try:
        print(f"[DEBUG] Requesting enhanced video: {video_id}")
        processed_path = get_processed_video_path(video_id)
        if not processed_path.exists():
            processed_path = get_processed_video_path(video_id, post_process=True)
        if not processed_path.exists():
            print(f"[ERROR] Enhanced video not found: {processed_path}")
            raise HTTPException(status_code=404, detail="增强后的视频不存在")
        
        print(f"[DEBUG] Serving enhanced video: {processed_path}")
        return FileResponse(
            path=str(processed_path),
            media_type="video/mp4",
            filename=f"enhanced_{video_id}.mp4",
            headers={
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-cache",
                "Access-Control-Allow-Origin": "*"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Unexpected error in download_video: {e}")
        raise HTTPException(status_code=500, detail=f"服务器错误: {str(e)}")


@router.websocket("/ws/progress/{video_id}")
async def ws_progress(websocket: WebSocket, video_id: str) -> None:
    """
    WebSocket：向前端推送实时进度/调度日志。

    当前实现为简单轮询 LOG 缓存并推送增量。
    """
    await websocket.accept()
    last_len = 0
    try:
        while True:
            logs = get_logs(video_id)
            if len(logs) > last_len:
                new_logs = logs[last_len:]
                await websocket.send_json({"logs": new_logs})
                last_len = len(logs)
            await asyncio.sleep(0.5)
    except WebSocketDisconnect:
        return


