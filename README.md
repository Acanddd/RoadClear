# RoadClear — 本机恢复版本

FastAPI + Vue 的天气感知道路视频增强与检测演示。当前默认使用 CPU，提供 AOD-Net、PReNet、TransWeather ONNX、天气分类、车辆检测和车牌检测。模型可加载不等于已复现论文指标；原训练和导出来源仍需补证。

## 当前电脑启动

在 PowerShell 7 中运行（已有独立环境和前端构建）：

```powershell
cd E:\acanhome\RoadClear
./scripts/check.ps1
./scripts/start.ps1
```

打开 <http://127.0.0.1:5173>，点击“启动系统”。选择 `samples/fog.mp4`、`samples/rain.mp4` 或 `samples/snow.mp4`，运行增强，再进入“任务评估”。这些是从保留的示例图片生成的 3 帧短视频，只验证功能闭环，不代表真实视频时序性能。

```powershell
./scripts/stop.ps1
```

启动脚本仅绑定本机，使用单个后端进程。启停信息在 `runtime/processes.json`，日志在 `runtime/logs/`；停止不会删除结果。如果端口占用或有旧进程记录，脚本会报错，不会自动终止其他程序。

## 新环境安装

要求：Windows、PowerShell 7、Python 3.12、Node.js >=22.13、pnpm 11.19.0。安装 pnpm 可使用 `npm install -g pnpm@11.19.0`。FFmpeg 由锁定的 `imageio-ffmpeg` 包提供，无需改系统 PATH。

1. 获取代码以及模型资产包。模型不提交 Git，也不会在启动时自动下载。按 `docs/model-manifest.json` 将六个文件放入 `models/`，用清单 SHA-256 核对。
2. 可复制 `.env.example` 为 `.env`；默认配置已适合本机。相对路径以项目根目录为基准。
3. 安装并构建：

```powershell
./scripts/setup.ps1 -PythonExe 'C:/path/to/python312/python.exe' -NodeExe 'C:/path/to/node.exe' -PnpmCmd 'C:/path/to/pnpm.cmd'
./scripts/start.ps1
./scripts/verify.ps1 -BaseUrl http://127.0.0.1:5173
```

工具已在 PATH 时可省略对应参数。`setup.ps1` 创建 `.venv-local`，安装 `backend/requirements.lock.txt` 和 `frontend/vue/pnpm-lock.yaml`，构建前端并校验模型。工具绝对路径仅存入忽略提交的 `runtime/toolchain.json`，源代码不依赖本机工具路径。

`backend/requirements.txt` 是直接依赖清单，`requirements.lock.txt` 是本次实际安装的完整 CPU 版本锁。修改依赖后应重新锁定并验证。当前仅在 Windows/Python 3.12 下实际完成安装；Linux 容器需继续验收。

## 目录和配置

| 位置 | 用途 |
|---|---|
| `backend/app/` | 当前后端运行代码 |
| `frontend/vue/` | 唯一正式前端，含锁文件 |
| `models/` | 六个模型文件；外部资产，不提交 Git |
| `runtime/` | 上传、结果、状态、参数、日志和验证输出 |
| `samples/` | 可提交的三个固定小样例 |
| `scripts/` | 安装、启停、环境和回归验证 |
| `docs/` | 模型、备份、归档清单与验收报告 |
| `archive/legacy/` | 本机保留的旧前端、训练脚本、第三方源码和旧资产，不再参与启动 |
| `backups/` | 修改前快照与初始 Git 元数据，不提交 Git |

主要环境变量见 `.env.example`。`ROADCLEAR_DISABLED_MODELS` 使用逗号分隔的 `aodnet,prenet,transweather,weather,vehicle,plate`。更改模型文件、设备或禁用列表后重启。`ROADCLEAR_BACKEND_URL` 配置 Vite 开发/预览代理目标，默认 `http://127.0.0.1:8000`；可在 `frontend/vue/.env.local` 中配置。网页请求与视频 URL 始终同源。

参数控制台仍可修改调度和后处理配置，保存到 `runtime/config/parameters.json`。经过验证的 ONNX/CPU 模型运行参数固定，相关前端控制禁用，后端拒绝改变它们。旧 `hdcwnet` 选择值和配置键保留兼容，新运行标识为 `transweather`。

## API 与失败行为

- `GET /health`：服务存活；`GET /ready`：全部六个模型就绪时 200，否则 503，并给出状态。
- `GET /models/status`：逐模型 `available/state/reason/device`；缺权重不阻止服务启动。
- `POST /video/upload_video`：分块存盘，验证类型、大小和视频内容，统一编码为 H.264 MP4。
- `POST /video/process/{video_id}`：推荐的处理入口；保留旧 GET 兼容接口并标记废弃。响应在处理完成后返回，不是异步任务队列。
- `GET /video/status/{video_id}`、`/video/download/{video_id}`：持久化状态与结果。重启中断的任务显示失败，需重试。
- `POST /eval/evaluate/{video_id}`：仅评估已成功处理的结果；车辆或车牌模型不可用则明确拒绝，不把缺失检测记为零。

缺模型返回 503，未知模型返回 400，无效 ID/视频返回 422，超限上传返回 413，不支持的格式返回 415，并发视频操作返回 409。处理失败清除本次结果和旧结果；临时编码写入非公开目录。不存在随机权重或原始帧静默兜底。

本机演示默认最多 100 MB、300 帧、最长边 1920 像素，且宽×高×帧数不超过 1.5 亿。处理、上传和评估互斥；不支持多进程部署、任务恢复或无人值守公网服务。输出保留帧率，但不保留原视频音轨。

TransWeather 当前 ONNX 在部分非方形输入上存在维度错误，适配器使用不超过 256 像素、64 倍数的方形输入，随后恢复原尺寸；这会改变缩放过程和小目标细节，不能将原论文数字直接归因于该部署版本。

## 容器部署（配置已交付，尚未实际运行验证）

先停止原生服务，安装并启动支持 Linux 容器的 Docker Desktop，再运行：

```powershell
docker compose config
docker compose up --build -d
./scripts/verify.ps1 -BaseUrl http://127.0.0.1:5173
docker compose restart
# 核对重启前视频 ID 的状态与下载结果
docker compose down
```

前端为 Nginx + Vue 构建产物，后端为 Python 3.12 CPU 镜像。`models/` 只读挂载，`runtime/` 持久化挂载，两端口都只暴露到 `127.0.0.1`。Compose 健康检查用 `/health`，因此部分模型缺失时仍可打开网页查看原因。

本机当前没有 Docker，`wsl --list` 也提示 WSL 尚未安装，不能把文件生成、YAML 解析或原生验证描述为容器运行成功。GPU 版 PyTorch/ONNX Runtime 也未安装或验证；即使本机有 NVIDIA GPU，默认仍为 CPU。

## 验证、恢复和证据

```powershell
./.venv-local/Scripts/python.exe scripts/audit_models.py
./scripts/verify.ps1
```

详细结果见 `docs/VALIDATION_REPORT.md`。环境输出、截图、逐视频结果保存在 `runtime/`。`scripts/browser-smoke.cjs` 是可选浏览器回归测试，需要 Playwright 和 Edge，默认不属于应用运行依赖。

备份位置和每个原文件的 SHA-256 记录在 `docs/backup-manifest.json`；归档映射见 `docs/archive-manifest.json`。恢复旧版时停止服务，把备份复制到一个新目录，避免直接覆盖本次结果。历史说明、论文和归档资产用于溯源，不作为当前部署验收证据。

Git 已初始化；大模型、视频运行产物、依赖目录、备份和本机配置均已排除。尚未代填 Git 姓名/邮箱，也未创建提交；配置自己的提交身份后即可提交当前恢复版本。
