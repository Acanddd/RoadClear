# 本机恢复验收报告

日期：2026-09-13。平台：Windows / Python 3.12 / CPU。报告描述当前恢复版本，不复用论文指标作为验收结果。

## 已验证

- 修改前备份：356 个文件，重新逐一核对备份 SHA-256 通过。备份和归档映射分别见 `backup-manifest.json` 与 `archive-manifest.json`。
- 六个模型：AOD-Net、PReNet、TransWeather、MobileNetV3 天气分类、DETRAC 车辆检测、CCPD 车牌检测，均成功加载并完成样例推理。模型形状、大小、哈希、耗时见 `model-manifest.json`。
- 独立 Python 3.12 环境从零安装成功；`pip check` 无依赖冲突。完整依赖已锁定。
- 前端从空目录按 pnpm 锁文件安装并构建成功；交付的 `setup.ps1` 也实际完整运行通过。
- 真实前端代理地址 `http://127.0.0.1:5173`：三种天气上传、增强、评估、结果下载和 HTTP Range 均通过。
- 每种天气重复处理并启用后处理；旧输出 URL 返回 404，新输出可下载。雪天旧选择值 `hdcwnet` 兼容通过。
- 实际 `stop.ps1` / `start.ps1` 重启通过；重启前已完成的视频状态仍为 completed，下载 Range 返回 206 / 100 字节。
- 10 项回归测试通过：格式/损坏/超限上传、缺模型与自动调度、失败后清理旧结果、非法 ID/模型、单操作互斥、状态持久化、无随机权重、无 GPU、全部资产与展示目录缺失时仍可启动。
- 无界面 Edge：通过页面选择样例、增强、实际播放原视频与结果视频、显示评估报告；页面无脚本异常。模拟 TransWeather 缺失后，三个依赖去雪的选项均禁用。

## 固定视频端到端记录

| 模型 | 结果 | 帧数 | 视频 FPS | 上传至结果校验耗时 |
|---|---|---:|---:|---:|
| aodnet | passed | 3 | 6.0 | 0.246 s |
| prenet | passed | 3 | 6.0 | 0.577 s |
| transweather | passed | 3 | 6.0 | 0.273 s |

这些视频是 128×96、3 帧的静态图片循环，耗时包含上传、转码、增强、下载及文件校验，不包含随后独立执行的检测评估。它们不是吞吐基准，不证明真实天气效果或时序稳定性。逐视频实际检测统计见 `validation-http.json`；没有标注数据，因此不报告 mAP/Precision/Recall。

## 修复的实际问题

- 禁止 AOD-Net/PReNet 加载失败后使用随机权重；天气分类缺权重也明确失败。
- 禁止逐帧异常回退原图并返回成功；输出先写非公开临时目录，完整验证后原子替换。
- 修复 Ultralytics 外层 `eval()` 导致进入训练流程的问题，改为底层网络 eval 和显式 predict。
- TransWeather 使用现有 ONNX；非方形输入广播错误通过 64 倍数方形预处理处理，CPU 输入上限为 256，再恢复原尺寸。
- 使用无界面 Matplotlib Agg，解决工作线程内 Tk 报错。
- 修复 `/health` 前端代理、静态图路径、视频 Range 支持、源视频帧率保留、持久化状态与 PowerShell 日期解析。
- 增强完成后自动将当前视频 ID 带入评估表单，避免跳转后评估按钮仍禁用。
- 旧展示页的置信度启发式指标已明确标为代理分数；加载失败时不展示预填的模拟指标。

## 尚未验证或待恢复

- **Docker 实机验收未完成**：本机没有 Docker，WSL 也尚未安装。Compose YAML 和引用文件的静态检查通过；镜像构建、Linux 依赖安装、容器健康检查、容器重启和容器内视频流程均未执行。安装并启动 Linux 容器运行环境后按 README 的容器命令验收。
- **GPU 加速未验证**：硬件为 RTX 3060 Laptop / 6 GB，但本次安装并验证的是 CPU 依赖。
- **原训练/导出来源待补证**：六个资产来自本机迁移快照，原始路径和哈希已记录，仍不能证明与论文最终 checkpoint 一致。TransWeather 的保留 PyTorch 预处理约定被沿用，但没有原 checkpoint 做数值一致性比较。
- **效果与性能边界**：TransWeather 方形缩放会影响细节；没有重新训练、GT 评测、全尺寸性能或真实视频质量验收。输出不保留音轨。
- 前端构建仍有约 1.17 MB 的主 JS chunk 警告，不影响此次本机功能验收。

## 本机产物

- `runtime/setup-final.log`：安装脚本执行输出。
- `runtime/pytest-final.log`：回归结果。
- `runtime/build-final.log`：构建结果。
- `runtime/environment.json`：环境和模型哈希核对。
- `runtime/browser-smoke.json`、`runtime/browser-smoke.png`、`runtime/browser-evaluation.png`：浏览器自动回归与截图。
- `runtime/validation-http.json`：真实 HTTP 验收记录。

本机配置、运行视频、报告截图和大模型不进入 Git。Git 仓库已初始化；因当前未配置提交身份，保持未提交状态。
