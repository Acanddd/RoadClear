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
    save_upload,
    video_info,
    validate_id,
    write_video,
)
from ..postprocessing import postprocess_frame

from ..operation import operation
from ..settings import DATA_DIR
import json
router = APIRouter()
STATUS_DIR = DATA_DIR / 'status'
STATUS_DIR.mkdir(parents=True, exist_ok=True)
def persist(video_id):
    path = STATUS_DIR / f'{validate_id(video_id)}.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(VIDEO_STATUS[video_id]), encoding='utf-8')
    temp.replace(path)


# 简单内存索引，记录视频是否已处理
VIDEO_STATUS: Dict[str, Dict[str, bool | str]] = {}


@router.post('/upload_video')
async def upload_video(file: UploadFile):
    with operation():
        video_id, path = await save_upload(file)
        VIDEO_STATUS[video_id] = {'processed': False, 'status': 'uploaded', 'frames_processed': 0}
        persist(video_id)
        return {'video_id': video_id}

@router.post('/process/{video_id}')
@router.get('/process/{video_id}', deprecated=True)
def process_video(video_id: str, model: str = 'auto', post_process: bool = False,
                  enable_multi_label: bool | None = None, multi_label_threshold: float | None = None):
    with operation():
        try:
            src = get_video_path(video_id)
        except FileNotFoundError:
            raise HTTPException(404, 'Video not found')
        for post in (False, True):
            get_processed_video_path(video_id, post).unlink(missing_ok=True)
        VIDEO_STATUS[video_id] = {'processed': False, 'status': 'processing', 'frames_processed': 0,
                                  'post_processed': post_process, 'model_chain': None}
        persist(video_id)
        dispatcher._video_states.pop(video_id, None)
        try:
            try:
                dispatcher.require_selection(model)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
            except RuntimeError as exc:
                raise HTTPException(503, str(exc)) from exc
            if enable_multi_label is not None or multi_label_threshold is not None:
                from ..config_manager import get_parameter_manager
                updates = {}
                if enable_multi_label is not None:
                    updates['enable_multi_label'] = enable_multi_label
                if multi_label_threshold is not None:
                    updates['multi_label_threshold'] = multi_label_threshold
                try:
                    get_parameter_manager().update_config({'dispatcher': updates})
                except ValueError as exc:
                    raise HTTPException(422, str(exc)) from exc
            info = video_info(src)
            VIDEO_STATUS[video_id]['total_frames'] = info['frames']
            dst = get_processed_video_path(video_id, post_process)
            chains = []
            def frames():
                for _, frame in iter_frames(src):
                    enhanced, _, chain = dispatcher.process_frame(frame, video_id, forced_model=model)
                    if post_process:
                        enhanced = postprocess_frame(enhanced)
                    if chain not in chains:
                        chains.append(chain)
                    VIDEO_STATUS[video_id]['frames_processed'] += 1
                    yield enhanced
                if VIDEO_STATUS[video_id]['frames_processed'] != info['frames']:
                    raise ValueError('Incomplete source decode')
            write_video(dst, frames(), fps=info['fps'])
            url = ('/postprocessed/' if post_process else '/processed/') + dst.name
            VIDEO_STATUS[video_id].update(processed=True, status='completed', model_chain='; '.join(chains), download_url=url)
            persist(video_id)
            return {'video_id': video_id, **VIDEO_STATUS[video_id], 'post_process': post_process}
        except Exception as exc:
            for post in (False, True):
                get_processed_video_path(video_id, post).unlink(missing_ok=True)
            VIDEO_STATUS[video_id].update(processed=False, status='failed', error=str(getattr(exc, 'detail', exc)))
            persist(video_id)
            add_log(video_id, f'Processing failed: {exc}')
            if isinstance(exc, HTTPException):
                raise
            raise HTTPException(500, f'Video processing failed: {exc}') from exc
        finally:
            dispatcher._video_states.pop(video_id, None)


def _stream_enhanced_frames(video_id: str, model: str = "auto", post_process: bool = False) -> Generator[bytes, None, None]:
    """
    以 multipart/x-mixed-replace 形式流式返回增强后视频（逐帧 JPEG）。
    """
    import cv2
    path = get_processed_video_path(video_id, post_process)
    for _, frame in iter_frames(path):
        ok, buf = cv2.imencode('.jpg', frame)
        if ok:
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n'

@router.get('/stream/{video_id}')
def stream_video(video_id: str, model: str = 'auto', post_process: bool = False):
    if not get_processed_video_path(video_id, post_process).is_file():
        raise HTTPException(409, 'Process the video before streaming')
    return StreamingResponse(_stream_enhanced_frames(video_id, model, post_process),
                             media_type='multipart/x-mixed-replace; boundary=frame')

@router.get("/status/{video_id}")
async def get_video_status(video_id: str) -> Dict[str, object]:
    """
    返回指定视频处理状态和进度信息。
    """
    validate_id(video_id)
    status = VIDEO_STATUS.get(video_id)
    if status is None:
        path = STATUS_DIR / f'{video_id}.json'
        if not path.is_file():
            raise HTTPException(404, 'Unknown video')
        status = json.loads(path.read_text(encoding='utf-8'))
        if status.get('status') == 'processing':
            status.update(status='failed', processed=False, error='Interrupted by restart; retry required')
    return {'video_id': video_id, **status}


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
