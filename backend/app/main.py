from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from . import settings, model_registry
from .utils.video_utils import VIDEO_DIR, PROCESSED_DIR, POSTPROCESSED_DIR

from .routes import config, evaluate, models, video, detection


def create_app() -> FastAPI:
    """
    创建并配置 FastAPI 应用实例。
    """
    app = FastAPI(
        title="RoadClear 后端服务",
        description="面向恶劣天气道路监控视频增强与评估的后端 API",
        version="0.2.0",
    )

    # 添加 CORS 支持，允许前端跨域访问
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    async def root():
        return {"message": "RoadClear backend is running."}

    # 兼容前端的健康检查
    @app.get("/health")
    async def health_check():
        return {"message": "RoadClear backend is healthy", "status": "alive"}

    @app.get("/ready")
    def ready():
        from fastapi.responses import JSONResponse
        caps = model_registry.capabilities()
        ready = all(caps.get(n, {}).get("available") for n in settings.MODEL_FILES)
        return JSONResponse({"ready": ready, "models": caps}, status_code=200 if ready else 503)

    # 挂载路由 - 必须在静态文件之前，确保动态路由优先匹配
    app.include_router(video.router, prefix="/video", tags=["video"])
    app.include_router(evaluate.router, prefix="/eval", tags=["evaluation"])
    app.include_router(models.router, prefix="/models", tags=["models"])
    app.include_router(config.router, prefix="/config", tags=["config"])
    app.include_router(detection.router, prefix="/detection", tags=["detection"])

    # 添加静态文件服务，让前端可以访问处理后的视频
    # 必须在所有路由之后挂载，避免被路由拦截
    videos_dir = VIDEO_DIR
    videos_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = PROCESSED_DIR
    processed_dir.mkdir(parents=True, exist_ok=True)
    postprocessed_dir = POSTPROCESSED_DIR
    postprocessed_dir.mkdir(parents=True, exist_ok=True)

    # 打印路径用于调试
    print(f"[INFO] Mounting /videos to: {videos_dir.absolute()}")
    print(f"[INFO] Mounting /processed to: {processed_dir.absolute()}")
    print(f"[INFO] Mounting /postprocessed to: {postprocessed_dir.absolute()}")

    app.mount("/videos", StaticFiles(directory=str(videos_dir)), name="videos")
    app.mount("/processed", StaticFiles(directory=str(processed_dir)), name="processed")
    app.mount("/postprocessed", StaticFiles(directory=str(postprocessed_dir)), name="postprocessed")

    # 挂载前端演示图片目录
    demo_images_dir = settings.DEMO_DIR
    if demo_images_dir.is_dir():
        app.mount("/demo_images", StaticFiles(directory=str(demo_images_dir)), name="demo_images")

    return app


app = create_app()
