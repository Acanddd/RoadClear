"""Bounded uploads and atomic browser-compatible video output."""
import uuid
from pathlib import Path
import cv2
import imageio.v2 as imageio
from fastapi import HTTPException
from ..settings import DATA_DIR, MAX_UPLOAD_BYTES

VIDEO_DIR = DATA_DIR / 'videos'
PROCESSED_DIR = DATA_DIR / 'processed'
POSTPROCESSED_DIR = DATA_DIR / 'postprocessed'
for directory in (VIDEO_DIR, PROCESSED_DIR, POSTPROCESSED_DIR):
    directory.mkdir(parents=True, exist_ok=True)

def validate_id(video_id):
    try:
        if str(uuid.UUID(video_id)) != video_id:
            raise ValueError()
    except (ValueError, AttributeError):
        raise HTTPException(422, 'Invalid video ID')
    return video_id

def get_video_path(video_id):
    path = VIDEO_DIR / f'{validate_id(video_id)}.mp4'
    if not path.is_file():
        raise FileNotFoundError(video_id)
    return path

def get_processed_video_path(video_id, post_process=False):
    directory = POSTPROCESSED_DIR if post_process else PROCESSED_DIR
    return directory / f'{validate_id(video_id)}.mp4'

def video_info(path):
    cap = cv2.VideoCapture(str(path))
    try:
        ok, frame = cap.read()
        if not cap.isOpened() or not ok:
            raise ValueError('Video is empty, corrupt, or unsupported')
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not 0 < fps <= 240:
            raise ValueError('Unsupported frame rate')
        h, w = frame.shape[:2]
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max(h, w) > 1920 or count <= 0 or count > 300 or h*w*count > 150_000_000:
            raise ValueError('Local demo limit: 1920px edge, 300 frames, 150 million decoded pixels; use a short clip')
        return {'fps': fps, 'frames': count, 'width': w, 'height': h}
    finally:
        cap.release()

def iter_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError('Cannot open video')
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield True, frame
    finally:
        cap.release()

def count_frames(path):
    return video_info(path)['frames']

def write_video(path, frames, fps=25.0):
    path = Path(path)
    temporary = DATA_DIR / 'tmp' / f'{uuid.uuid4()}.writing.mp4'
    temporary.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with imageio.get_writer(str(temporary), fps=fps, codec='libx264', pixelformat='yuv420p', macro_block_size=1,
                                ffmpeg_params=['-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-movflags', '+faststart']) as writer:
            for frame in frames:
                writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                count += 1
        if count == 0 or not temporary.is_file():
            raise ValueError('No frames produced')
        info = video_info(temporary)
        if info['frames'] != count:
            raise ValueError('Encoded frame count mismatch')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

async def save_upload(file):
    ext = Path(file.filename or '').suffix.lower()
    if ext not in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
        raise HTTPException(415, 'Unsupported video extension')
    if file.content_type and not (file.content_type.startswith('video/') or file.content_type == 'application/octet-stream'):
        raise HTTPException(415, 'Expected a video upload')
    video_id = str(uuid.uuid4())
    raw = VIDEO_DIR / f'{video_id}.upload{ext}'
    final = VIDEO_DIR / f'{video_id}.mp4'
    try:
        size = 0
        with raw.open('wb') as output:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, 'Video exceeds upload size limit')
                output.write(chunk)
        from starlette.concurrency import run_in_threadpool
        def normalize():
            info = video_info(raw)
            write_video(final, (frame for _, frame in iter_frames(raw)), fps=info['fps'])
            if video_info(final)['frames'] != info['frames']:
                final.unlink(missing_ok=True)
                raise ValueError('Incomplete video decode')
        await run_in_threadpool(normalize)
        return video_id, final
    except HTTPException:
        raise
    except Exception as exc:
        final.unlink(missing_ok=True)
        raise HTTPException(422, f'Invalid video: {exc}') from exc
    finally:
        raw.unlink(missing_ok=True)
        await file.close()
