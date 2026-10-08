# RoadClear 项目掌握手册

> 目标：把毕业设计从“做过、能运行过”变成“能解释、能复现、能质疑、能继续开发”。
>
> 本手册基于 `paper.pdf`、当前源码、模型资产、训练结果和前端构建检查整理。论文中的实验结论与当前仓库的可运行状态并不完全等价，二者在下文会明确区分。

## 1. 先用一句话说清项目

RoadClear 是一个面向恶劣天气道路监控视频的任务驱动增强系统：它使用轻量级 MobileNetV3-Small 判断雾、雨、雪，根据天气动态调度 AOD-Net、PReNet 或 TransWeather 等专用恢复模型，再用车辆/车牌检测和图像质量指标评估增强是否真正改善下游视觉任务，并通过 Vue + FastAPI 提供视频处理、参数控制和报告展示。

英文版：

> RoadClear is a task-oriented video enhancement system for road surveillance under fog, rain, and snow. It classifies weather with MobileNetV3-Small, dispatches weather-specific restoration models, and evaluates whether enhancement improves downstream vehicle and licence-plate detection through a Vue and FastAPI application.

不要把项目说成“我提出了 AOD-Net/PReNet/TransWeather 的网络结构”。这些网络来自已有研究；但你可以明确认领：基于 Snow100K 自行训练了项目专用的 TransWeather 去雪权重，并完成了它在视频增强系统中的适配、调度和评估。你的其他主要贡献是系统集成、天气感知调度、实验设计、任务级评估、工程实现，以及基于实验得到的分析结论。

## 2. 项目到底解决什么问题

传统做法常把“图像看起来更清晰”当成终点，但道路监控真正关心的是车辆、车牌等目标是否更容易被检测。RoadClear 因此同时处理两个问题：

1. 不同天气退化机制不同，单个恢复模型难以跨雾、雨、雪泛化。
2. PSNR、SSIM 或噪声等像素指标变好，不保证目标检测也变好；反过来也可能成立。

你的核心研究假设是：

- 先识别天气，再选择对应的专用模型，会优于不分天气地使用同一个恢复模型。
- 恢复效果应该同时由像素质量、下游检测表现和运行性能评估。
- 视频相邻帧的天气通常变化缓慢，因此不需要每帧重复分类。

## 3. 系统数据流

```text
上传视频
   ↓
保存并必要时转码为 MP4/H.264
   ↓
逐帧读取
   ↓
每 N 帧调用 MobileNetV3-Small，其他帧复用上次天气状态
   ↓
fog  → AOD-Net
rain → PReNet
snow → 基于 Snow100K 自行训练的 TransWeather 专用权重
混合天气 → 按概率排序的模型级联（带 fog 优先规则）
   ↓
可选 CLAHE / 锐化 / 降噪 / 动态范围后处理
   ↓
重新编码视频并提供播放/下载
   ↓
YOLO 车辆检测 + YOLO 车牌检测 + 图像/时序指标
   ↓
Vue 页面展示进度、日志、对比结果和 HTML 报告
```

当前代码入口与职责：

| 模块 | 作用 | 你必须理解的内容 |
|---|---|---|
| `backend/app/main.py` | FastAPI 入口与路由/静态目录挂载 | ASGI、路由、CORS、静态文件 |
| `backend/app/routes/video.py` | 上传、处理、状态、流式播放、下载、WebSocket | 请求生命周期、生成器、同步计算阻塞 |
| `backend/app/scheduler/weather_classifier.py` | MobileNetV3 三分类推理 | 输入归一化、logits、softmax、类别映射 |
| `backend/app/scheduler/dispatcher.py` | 跳帧分类、模型选择、级联、模型池、FPS 统计 | 状态缓存、调度规则、失败回退 |
| `backend/app/models/*.py` | 恢复模型适配层 | BGR/RGB、张量布局、归一化、权重和 ONNX provider |
| `backend/app/evaluation/detector.py` | 车辆与车牌检测 | YOLO 推理、置信度阈值、边界框 |
| `backend/app/evaluation/metrics.py` | 无 GT 的在线指标 | 指标定义及其局限性 |
| `backend/app/routes/evaluate.py` | 原始/增强视频比较与 HTML 报告 | 评估数据流、内存占用、FPS 含义 |
| `backend/app/config_manager.py` | JSON 参数持久化与 Pydantic 校验 | 配置模型、全局状态、并发风险 |
| `frontend/vue/src/components/*` | 上传、增强、参数、算法展示、评估 | Vue Composition API、fetch、轮询 |

## 4. 研究工作拆解

### 4.1 天气分类

- Backbone：MobileNetV3-Small。
- 输入：RGB，`224 × 224`。
- 输出：fog、rain、snow 三类 softmax 概率。
- 训练集：论文称 RSCM 每类 10,000 张。
- 外部测试集：DAWN，共 737 张，fog/rain/snow 支持数分别为 335/200/202。
- 训练策略：加权交叉熵、AdamW、余弦退火、数据增强、早停；代码默认训练/验证比为 80/20，并固定 NumPy seed 为 42。

论文报告的 DAWN 单标签结果：

| 类别 | Precision | Recall | F1 |
|---|---:|---:|---:|
| Fog | 80.11% | 84.18% | 82.10% |
| Rain | 72.84% | 59.00% | 65.19% |
| Snow | 86.55% | 95.54% | 90.82% |

最值得解释的是 rain recall 只有 59.00%，其中相当一部分雨天被判成雾。论文把原因解释为雨雾共存和视觉退化相似。

当前所谓“多标签”不是重新用 sigmoid + BCE 训练的真正多标签模型，而是对三分类 softmax 的多个概率做 `> 0.3` 阈值判断。它提高“真实单标签被候选集合包含”的概率，但候选变多本来就更容易提高 recall。因此可以称为“threshold-based multi-weather inference heuristic”，不要在面试中说成“训练了多标签分类网络”。论文报告：rain recall 由 59.00% 提升到 75.50%，但这个数字必须连同上述口径一起解释。

### 4.2 跳帧调度

调度器为每个视频保存：已处理帧数、上一次天气、天气描述和模型链。配置默认每 30 帧重新分类一次，其余帧复用结果。这利用了天气的时间连续性，降低分类开销。

你要能回答三个问题：

- 为什么是 30？当前它是经验超参数，不是论文中充分消融得到的最优值。
- 天气突然变化怎么办？最坏会延迟约 30 帧才切换；应通过自适应采样、置信度触发或场景变化检测改善。
- 为什么需要 hysteresis/EMA？防止概率在边界附近波动造成模型频繁切换和视频闪烁。当前 EMA 只用于 FPS 统计，没有平滑天气概率。

### 4.3 专用恢复模型

| 天气 | 论文/设计模型 | 机制 | 你的贡献边界 |
|---|---|---|---|
| Fog | AOD-Net | 把大气散射模型重写为端到端映射 | 集成、权重/ONNX 转换、预后处理与调度 |
| Rain | PReNet | 通过共享参数的 recurrent stages 逐步去雨 | 集成、ONNX 适配、迭代次数配置与调度 |
| Snow | 基于 Snow100K 自行训练的 TransWeather 专用权重 | Transformer 型统一天气恢复 | 训练/选择最终去雪权重、系统适配、调度与评估；HDCWNet 是训练过程中因视频流效果不佳而弃用的候选方案 |

模型输入输出适配是实际工程工作的一部分：OpenCV 使用 BGR，模型一般使用 RGB；NumPy 图像是 `HWC uint8`，PyTorch/ONNX 通常要求 `NCHW float32`；不同模型又使用 `[0,1]` 或 `[-1,1]` 归一化。错误的通道顺序或归一化会让程序“能跑但结果完全不可信”。

### 4.4 下游任务与评估

论文使用 UA-DETRAC 训练四类车辆检测器，并在干净视频上合成雨、雾、雪退化。测试规模称为 20 个场景、每个 5 秒、25 FPS，共 2,500 帧。CCPD 用于车牌检测。

评估分三层：

- 像素层：PSNR、SSIM、对比度、噪声抑制率。
- 任务层：mean confidence、高置信度检测数、mAP、DCCR、Recovery Rate、按目标尺寸划分的 Recall。
- 性能层：FPS、参数量、时序稳定性。

论文最有说服力的结果：

- 雾：AOD-Net 将小目标 Recall 从 0.4667 提升到 0.6123，即相对提升 31.19%。
- 雨：PReNet 将 mAP@0.95 从 0.0736 提升到 0.1680，即相对提升 128.18%；绝对提升是 0.0944。面试时必须同时给出基线，避免只报一个夸张百分比。
- 雪：TransWeather 将高置信度检测数从 596.4 提升到 650.1，约 9.01%；大目标 Recall 提升 10.32%，小目标仅提升 3.16%。
- 跨天气消融：专用模型在对应天气上最好；例如 AOD-Net 用于雪天时，论文称 mAP@0.95 从 0.249 降到 0.062，下降 75.0%。

### 4.5 最值得面试深入谈的研究发现

雾天增强后，论文中的噪声抑制率为 `-234.86%`，但检测指标反而明显提高。这说明拉普拉斯方差等高频统计把增强后出现的边缘和纹理也当成了“噪声”，而这些高频结构可能正是检测器需要的语义信息。

你可以这样表达：

> I found that pixel-level restoration metrics did not always correlate with downstream detection. Under fog, AOD-Net increased high-frequency energy and therefore produced a negative noise-suppression score, yet small-object recall improved by 31.19%. This led me to evaluate enhancement as a task-oriented pipeline rather than relying on visual quality alone.

这个结论是合理的观察，但“pixel-semantic discrepancy”是你对现象的命名，不应说成已经被广泛验证的理论。要进一步成立，需要更多数据集、不同检测器、相关性分析和统计显著性检验。

## 5. 哪些内容可以算你的贡献

可以合理认领：

- 定义恶劣天气视频的完整处理与评估问题。
- 训练/评估 MobileNetV3-Small 天气分类器（前提是你能复现实验并解释训练代码）。
- 设计并实现跳帧天气分类、模型选择、手动/自动模型链和 per-video state。
- 将异构恢复网络封装为统一 `enhance(frame)` 接口。
- 基于 Snow100K 训练项目专用 TransWeather 去雪权重，并在实验比较后弃用视频流效果不佳的 HDCWNet 候选方案。
- 将 PyTorch 模型转换/接入 ONNX Runtime 的 AOD-Net、PReNet 推理路径。
- 训练或微调车辆/车牌 YOLO 检测器，并搭建增强前后评估流程（需明确车牌是 detection，不是 OCR recognition）。
- 设计天气专用模型消融与下游任务评估。
- 实现 Vue + FastAPI 的上传、处理、参数配置、播放、日志和报告界面。

不要过度认领：

- 不要说“提出/发明 AOD-Net、PReNet 或 TransWeather 的网络结构”；可以说“trained a task-specific TransWeather model on Snow100K and integrated it into the pipeline”。
- 不要仅根据当前迁移快照判断原项目的全部部署状态；当前文件显示 AOD-Net/PReNet 的 ONNX 路径和 Ultralytics `.pt` 检测路径，原项目是否所有模型都走 ONNX 应以保留的实验日志或原环境为准。
- 当前不能说“production-ready”或“cloud deployed”。
- 在没有可靠端到端基准前，不要说“实现实时 25 FPS”。
- `LicensePlateDetector` 只检测车牌框；CRNN 历史显示验证准确率接近 0，不能称为可靠车牌字符识别系统。

## 6. 当前仓库审计：原始项目与迁移资产快照要分开

当前目录是从远程环境迁移后的项目资产快照。为控制体积，部分大权重、第三方源码、数据集和过程文件已经主动删除，因此“当前目录不能直接部署运行”不等于“毕业设计当时没有训练成功或没有运行过”。下面的审计分为两类：

- **原始项目事实**：由论文、保留的实验结果和你的开发记录支持，例如 Snow100K 训练的 TransWeather、论文实验和演示系统。
- **当前快照状态**：用于判断现在要怎样恢复可复现环境，不能反向否定已经完成的原始实验。

### 6.1 已有且较扎实的资产

- 49 页完整论文，研究问题、相关工作、方法、实验、消融和系统界面齐全。
- FastAPI API、Vue 页面和参数控制已经形成较完整交互面。
- 当前仍保留天气分类权重、AOD-Net/PReNet/TransWeather ONNX 资产、车辆/车牌 YOLO 权重和部分训练结果；其他大文件和过程文件已经在迁移时删除。
- Vue 生产构建可以成功，当前生成包约 1.17 MB（gzip 约 375 KB），但存在大 chunk 警告。
- 车辆 YOLO 训练结果文件显示最终验证 mAP50 约 0.916、mAP50-95 约 0.701；车牌检测结果显示 mAP50 约 0.891、mAP50-95 约 0.251。需先核对 CSV 表头再在简历使用。

### 6.2 当前快照与原始项目之间需要补齐的迁移信息

1. **雪天命名保留了历史包袱**：最终方案是 Snow100K 训练的 TransWeather；HDCWNet 已经因视频流效果不好而弃用。当前 `dispatcher.py` 仍通过名为 `HDCWNetEnhancer` 的兼容类加载 TransWeather，容易让读者误以为最终模型是 HDCWNet，应重命名为 `TransWeatherEnhancer` 并在文档记录模型选择过程。
2. **当前快照无法重建原始雪天路径**：代码寻找已删除的 `TransWeather-main/epoch_90`，同时保留了 `transweather_desnow.onnx`。这更准确地说明迁移后缺少原始权重/源码映射，而不是原项目没有完成雪天实验。恢复时应以最终 TransWeather 权重的 checksum、训练配置和推理入口为准。
3. **当前 Docker 文件不是完整迁移包**：根目录没有 compose 文件，镜像也没有包含所有已删除的大权重、外部模型和演示资产。它只能代表当时部署脚本的残片，不能直接用于重建；需要重新制作轻量部署清单和权重下载/挂载机制。
4. **ONNX 路径需要重新核对**：当前快照显示 AOD-Net/PReNet ONNX、TransWeather ONNX 和 Ultralytics `.pt` 检测器等多种路径。简历中应只写能够从原日志或新复现实验确认的加速范围。
5. **“多标签分类”口径需要收紧**：现有网络仍是 softmax 单标签训练，只是在推理时允许多个 softmax 概率超过阈值。
6. **车牌识别实际上主要是检测**：在线评估统计的是车牌框数量，不是字符识别准确率；CRNN 训练历史并不支持可用的 OCR 声明。
7. **前端演示卡有硬编码指标**：`YoloComparisonCard.vue` 中 mAP、Precision、Recall、F1 的部分数值是常量，不应当作实时 API 计算结果展示或用于项目证明。

### 6.3 工程风险

- `/process/{video_id}` 是 `async` 路由，但内部执行同步逐帧深度学习和编码；单 worker 下可能阻塞事件循环，使状态轮询/WebSocket 在处理期间无法及时响应。
- 上传接口一次性 `await file.read()`，大视频会整体进入内存，且缺少大小、MIME、扩展名和内容校验。
- `VIDEO_STATUS`、日志、调度状态和模型池都在进程内存中；重启丢失，多 worker 不一致。
- 配置存入一个全局 JSON 文件，用户请求还能修改全局 dispatcher 配置，存在并发覆盖和实验不可追溯问题。
- 评估把视频所有帧和检测结果留在列表中，长视频会显著占用内存。
- CORS 使用通配源并允许 credentials，部署时应改成明确前端域名。
- 模型在模块导入时全局初始化，启动慢、故障影响面大，也难以做健康检查和按需扩缩容。
- 依赖没有完全锁定：`torch`、`torchvision` 没版本，Docker base 使用 `latest`，Python/TensorFlow/CUDA 兼容性不可复现；代码还使用未列入 requirements 的 `imageio`。
- 仓库没有顶层 README、`.gitignore`、许可证、数据/权重获取脚本和实验 manifest；还提交了 `node_modules`、缓存、视频、权重和大量第三方仓库。
- 当前 `.venv` 指向一个不存在的 Python 3.14 安装，不能作为可复现环境。

### 6.4 研究有效性风险

- DETRAC 同时用于训练车辆检测器和生成合成天气测试集。必须确认按原始视频 sequence 做 train/test 隔离，否则相邻帧或同场景泄漏会高估结果。
- 论文重点结果来自合成天气；真实雨雾雪视频的外部有效性仍有限。
- 在线 `evaluate.py` 的 mean confidence、检测数和时序稳定性是无 Ground Truth 指标，检测更多也可能意味着更多假阳性，不能替代 precision/recall/mAP。
- 代码里的 temporal stability 实际只基于逐帧检测数量的变异系数，并没有匹配同一目标的轨迹，也不能直接证明画面无闪烁。
- noise suppression 使用拉普拉斯方差，高频纹理和噪声不可区分；这既解释了雾天异常，也说明该指标本身有限。
- 缺少置信区间、重复实验、显著性检验和不同 detector 的交叉验证。
- 论文称所有模型都经 ONNX 加速、系统可实时部署，但没有给出完整的逐模型 latency、吞吐、显存、warm-up 与端到端 FPS 表。

## 7. 简历写法

### 7.1 面向 Computer Vision / ML Internship

项目名：**RoadClear — Weather-Aware Road Surveillance Enhancement System**

- Built a weather-aware video enhancement pipeline that classifies fog, rain, and snow with MobileNetV3-Small and dispatches dedicated AOD-Net, PReNet, and TransWeather restoration models; trained a task-specific TransWeather checkpoint on Snow100K after discarding HDCWNet due to poor video-stream quality.
- Designed task-oriented evaluation across synthetic adverse-weather DETRAC videos, combining image-quality metrics with YOLO detection metrics and weather-specific ablations.
- Observed a pixel/semantic metric mismatch under fog: dehazing improved small-object recall by 31.19% despite a negative high-frequency noise-suppression score.
- Integrated model inference, video processing, parameter controls, and evaluation reports in a Vue/FastAPI application; converted AOD-Net and PReNet inference paths to ONNX Runtime.

使用前提：把 Snow100K 训练配置、最终 checkpoint/checksum、关键曲线和推理对比证据补回项目档案。简历可以描述毕业设计当时已完成的训练和实验，但 GitHub/作品集 README 应注明公开仓库是删除大文件后的轻量快照，并提供恢复方式。

### 7.2 面向 Backend / Software Engineering Internship

- Developed a FastAPI service and Vue 3 interface for video upload, frame-wise ML processing, configurable model routing, progress/status reporting, streaming playback, and HTML evaluation reports.
- Implemented a unified model-adapter interface and a stateful dispatcher that reuses weather predictions across frames and supports automatic or manually composed enhancement chains.
- Built a multi-stage evaluation workflow comparing original and enhanced video with vehicle/plate detection, confidence, temporal, and image statistics.
- Containerised an initial prototype and identified the next production architecture: background job workers, PostgreSQL job metadata, object storage, versioned model registry, and GPU-aware deployment.

这里不要写 “deployed with Docker”——当前只能诚实写 “containerised an initial prototype” 或等完整 compose 和镜像通过后再升级措辞。

### 7.3 中文简历压缩版

- 构建面向雾、雨、雪道路监控的天气感知视频增强系统，以 MobileNetV3-Small 分类天气并动态调度专用图像恢复模型。
- 基于合成恶劣天气 DETRAC 视频设计图像质量、车辆检测与跨天气消融评估；雾天小目标 Recall 相对提升 31.19%。
- 使用 FastAPI、Vue 3、OpenCV、PyTorch/ONNX Runtime 实现视频上传、逐帧推理、参数配置、结果播放与评估报告。

## 8. 面试讲述模板

### 8.1 30 秒版

我的毕业设计 RoadClear 解决恶劣天气下道路监控目标检测退化的问题。系统先用 MobileNetV3-Small 判断雾、雨、雪，再选择 AOD-Net、PReNet 或 TransWeather 做专用恢复，并用 YOLO 的检测效果而不只是 PSNR/SSIM 来判断增强是否有价值。我负责整体流水线、天气调度、模型适配、评估实验以及 Vue/FastAPI 系统。最有意思的发现是雾天像素噪声指标变差，但小目标 Recall 反而提升了 31.19%，说明视觉增强必须结合下游任务评估。

### 8.2 两分钟版结构

1. **Problem**：恶劣天气降低监控画质和车辆检测表现；单模型跨天气泛化差。
2. **Design**：轻量天气分类 + 跳帧调度 + 三个天气专用恢复模型 + YOLO 下游评估。
3. **Engineering**：统一模型接口，处理颜色/张量/归一化差异；FastAPI 管视频任务，Vue 做参数和结果展示；AOD/PReNet 接入 ONNX。
4. **Experiment**：RSCM→DAWN 测分类，DETRAC 合成三种天气测增强与检测，做跨模型消融。
5. **Result**：雾天小目标、雨天严格 IoU 指标、雪天大目标均有不同程度恢复；错误模型会明显伤害性能。
6. **Reflection**：当前研究不足是合成数据和统计验证有限；迁移后的工程快照还缺少完整权重清单与可复现环境。下一步是恢复轻量可复现部署，并加入数据库/任务队列和真实天气外部测试。

### 8.3 AI 辅助开发该怎么诚实回答

推荐回答：

> I used AI tools to accelerate boilerplate, debugging, and integration, especially across FastAPI, Vue, and model wrappers. I did not treat generated code as proof of correctness. I traced the data flow, checked tensor shapes and preprocessing, compared model outputs, and designed the experiments around downstream detection. During my later audit I also found gaps between the paper and repository, such as the incomplete snow-model deployment path and hard-coded demo metrics. I am now fixing those systematically and can explain the trade-offs and limitations rather than only demonstrating the UI.

重点不在于否认 AI，而在于展示你有代码所有权：能画数据流、解释每个接口、复现实验、定位失败、指出局限并亲自修改。

## 9. 高频深挖问题与回答要点

### 为什么选 MobileNetV3-Small？

它使用 depthwise separable convolution、inverted residual 和 SE 模块，在较低参数量与计算量下保持合理精度，适合作为每隔若干帧运行的前置分类器。需要补充：你没有做过充分 backbone latency/accuracy 对比，所以应说是基于轻量化目标的工程选择，而不是已证明全局最优。

### 为什么不同天气要不同模型？

雾主要是全局对比度衰减和大气散射，雨是局部方向性条纹，雪更多是局部遮挡。它们的退化机制不同；论文的跨天气消融显示错配模型会降低检测性能，尤其去雾模型在非雾图像上可能过度改变全局对比度。

### softmax 为什么还能多标签？

严格来说它不是真正多标签学习，而是从互斥三分类概率中取多个超过阈值的候选，作为混合天气调度 heuristic。更严谨的改进是构造多天气标注，用三个独立 sigmoid 和 BCE/Focal loss 训练，并按 label-wise precision/recall、mAP 和 calibration 评估。

### ONNX 为什么可能更快？

它把训练框架图转换为推理图，由 ONNX Runtime 做常量折叠、算子融合、内存规划并选择 CUDA/CPU Execution Provider。必须分别报告 warm-up 后的 median/P95 latency、batch size、输入分辨率和 provider；不能只凭“转换成功”声称加速。

### 你的 mAP@0.95 提升 128.18% 是否夸大？

这是相对提升，因为基线只有 0.0736，增强后是 0.1680；绝对提升 0.0944。这个结果说明严格定位指标明显恢复，但最终数值仍不高。主动解释相对与绝对变化会增强可信度。

### 为什么检测数量增加不一定是好事？

没有 GT 时无法区分新增 true positives 和 false positives。因此线上报告里的检测数、confidence、DCCR 只是 proxy；研究结论应以有标注测试集上的 precision/recall/mAP 为主。

### 你所谓 temporal stability 测了什么？

当前实现是 `1 / (1 + std(counts)/mean(counts))`，只衡量每帧检测数量波动。它不跟踪同一辆车，也不直接测图像闪烁。更好的方法是用 tracker 建立轨迹，计算 box IoU/置信度抖动，或对相邻增强帧做光流对齐后的 temporal warping error。

### 系统为什么目前不适合多用户？

任务状态和日志在进程内存，全局 JSON 配置会被请求共同修改，长任务还在 API 进程同步执行。多 worker 会状态不一致，重启会丢任务。应把任务元数据放 PostgreSQL，队列放 Redis，处理放独立 GPU worker，视频放对象存储。

### 最难的工程问题是什么？

可以从以下真实问题中选一个你亲自复现后讲：异构模型预处理一致性、模型权重/路径管理、PyTorch→ONNX 数值一致性、视频编码兼容、长任务进度、混合天气的级联顺序。不要只说“调参很难”。

## 10. 数据库与异步处理升级设计

数据库不应该存视频二进制或模型权重。视频/报告放 S3、MinIO 或本地对象目录；PostgreSQL 存元数据和可查询结果。

建议最小表结构：

```text
videos
- id UUID PK
- original_name
- object_uri
- codec / width / height / fps / frame_count
- file_size / sha256
- created_at

processing_jobs
- id UUID PK
- video_id FK
- status: queued/running/succeeded/failed/cancelled
- requested_model_chain
- config_snapshot JSONB
- progress_frames / total_frames
- worker_id / error_message
- started_at / finished_at

model_versions
- id UUID PK
- name / task / framework
- version / weights_uri / checksum
- preprocessing JSONB
- metrics JSONB
- active

inference_runs
- id UUID PK
- job_id FK
- model_version_id FK
- weather_prediction JSONB
- latency_ms / device / provider
- output_uri

evaluation_runs
- id UUID PK
- job_id FK
- evaluator_version
- metrics JSONB
- report_uri
- created_at
```

推荐请求流程：

```text
POST /videos → 流式上传对象存储 → 创建 video
POST /jobs   → PostgreSQL 写 queued → Redis 入队 → 立即返回 202 + job_id
GPU worker   → 拉取视频 → 推理 → 周期更新 progress → 上传输出
GET /jobs/id → 读 PostgreSQL/缓存
WebSocket/SSE → 推送进度
GET /evaluations/id → 指标和报告
```

这样数据库带来的不是“为了连接数据库而连接”，而是任务持久化、实验可追溯、多用户隔离、失败重试和可查询的模型版本。

## 11. 部署与工程化路线

### P0：1-2 周，先让项目可复现

- 新建干净 Git 仓库；补顶层 README、架构图、演示 GIF、许可证和 `.gitignore`。
- 移除 `node_modules`、`.venv`、`__pycache__`、运行视频和重复第三方源码；用脚本下载公开权重与数据。
- 建立 `third_party/` 与 `NOTICE`，记录 AOD-Net/PReNet/TransWeather 的来源和许可证。
- 恢复 Snow100K→TransWeather 的训练配置、最终指标、checkpoint checksum 和 ONNX 导出记录；将兼容类 `HDCWNetEnhancer` 正式重命名为 `TransWeatherEnhancer`，并记录 HDCWNet 被弃用的实验依据。
- 删除或明确标注硬编码 demo 指标，让 UI 从真实评估结果读取。
- 锁定 Python、CUDA、PyTorch、ONNX Runtime、Node 版本；生成可重建 lockfile。
- 写 smoke test：加载三个天气模型、处理一帧、检查尺寸/类型/有限值；再跑一个 10 帧视频 E2E。

完成标准：新机器按 README 能启动，三个天气路径都能处理固定样例，输出有校验值或可接受容差。

### P1：1-2 周，修成长任务后端

- 把同步视频处理移到 Celery/RQ/Dramatiq 或自建 worker。
- PostgreSQL + Alembic 管视频、任务、配置快照、模型版本和结果。
- Redis 负责队列和短期进度；对象存储负责视频、权重、报告。
- 上传改成流式写入，限制大小/类型；加入任务取消、超时、重试和幂等键。
- API 使用 202 Accepted，任务状态遵循明确状态机。

### P2：1 周，完成真正的容器部署

- Vue 使用 multi-stage Node build，产物交给 Nginx。
- FastAPI 使用固定 Python/CUDA base；模型权重通过只读 volume 或对象存储加载。
- 建立 `compose.yaml`：`nginx/frontend + api + worker + postgres + redis + minio`；GPU worker 使用 NVIDIA Container Toolkit。
- 增加 readiness/liveness；模型未加载时 readiness 失败但进程日志要清晰。
- 配置全部来自环境变量/secret，不把路径和密钥写死。

云上演示最现实的方案是单台 GPU EC2/云 GPU 主机运行 compose；流量和稳定性提升后再拆 ECS/EKS。不要为了简历一开始就过度设计 Kubernetes。

### P3：1-2 周，测试与 CI

- `pytest` 单测：指标边界值、配置校验、调度映射、颜色/形状转换。
- 集成测试：使用 fake enhancer/fake detector，不加载大型权重也能测试 API 和任务状态。
- golden test：固定输入帧、模型版本和输出统计，检测预处理/模型更新造成的漂移。
- 前端组件测试和最小 Playwright E2E。
- GitHub Actions：lint、type check、unit test、Vue build、Docker build；GPU 测试单独手动或定时运行。

### P4：2-4 周，提升研究可信度

- 按 DETRAC 原始 sequence 分组切分训练/验证/测试，明确防止相邻帧泄漏。
- 加入真实恶劣天气外部测试集，并分别报告 synthetic 与 real-world 结果。
- 基于 GT 统一计算 mAP50、mAP50-95、precision、recall 和 size-specific AP/AR。
- 对 20 个场景报告均值、标准差/95% CI，并进行 paired significance test。
- 比较至少两个 detector，验证结论是否依赖某个 YOLO 权重。
- 增加 no-enhancement、wrong-model、single-model、oracle-weather、predicted-weather 五组消融，隔离分类错误造成的损失。
- 报告端到端 latency：分类、恢复、检测、编码分别计时，给出 warm-up、P50/P95、FPS、显存与功耗。

### P5：研究改进方向

- 真正的 multi-label weather model：sigmoid + BCE/Focal loss + 混合天气标注与概率校准。
- 调度稳定性：天气概率 EMA、hysteresis、scene-change trigger、自适应分类间隔。
- 时序恢复：光流对齐、recurrent/temporal transformer，尤其针对雪遮挡和小目标信息丢失。
- 检测感知增强：将 detector feature/loss 纳入训练目标，但要防止过拟合某个检测器。
- ROI/分辨率策略：远处小目标区域使用更高分辨率恢复，背景区域轻量处理。
- TensorRT/FP16、异步解码/编码、batching 和 CUDA stream，用真实 profiler 决定优化点。

## 12. 建议的掌握顺序

不要一开始重写全部项目。按下面顺序，每一阶段都要能脱离 AI 讲解并做小改动。

### 第 1 周：画懂与跑通

- 手画从 upload 到 output/report 的调用图。
- 解释 BGR↔RGB、HWC↔NCHW、归一化和 `no_grad()`。
- 跑固定图片的天气预测、AOD-Net、PReNet；记录输入输出 shape、dtype、范围和 latency。
- 从迁移资产重建并验证最终 TransWeather snow path，同时保留 HDCWNet→TransWeather 的模型选择记录。

### 第 2 周：掌握模型与指标

- 能从零写 softmax、cross-entropy、precision/recall/F1、IoU、AP/mAP 的简化版本。
- 读懂 MobileNetV3 的 depthwise convolution、inverted residual、SE。
- 读懂 AOD-Net 的成像假设、PReNet 的 recurrence、Transformer 去天气的基本思路。
- 用一个小样例亲手计算 DCCR、Recovery Rate 和当前 temporal stability。

### 第 3 周：掌握后端

- 用小型 fake model 重写一次 upload→job→result。
- 理解为什么 `async def` 内部跑 CPU/GPU 同步任务仍会阻塞。
- 接 PostgreSQL，完成 video/job/model/evaluation 四类最小实体和 Alembic migration。
- 将处理迁移到 worker，支持失败状态与重试。

### 第 4 周：掌握部署与实验复现

- 写可工作的 compose；在干净环境启动。
- 建立固定实验配置、随机种子、dataset split manifest、模型 checksum。
- 复现论文中至少一项主要结果，并保存命令、日志、原始 CSV 和图表。
- 录制 2 分钟演示，并进行一次不看稿项目讲述。

## 13. “我真的掌握了”的验收清单

只有当下面问题大部分都能现场回答/操作时，才算真正拥有这个项目：

- [ ] 五分钟内从上传讲到评估报告，能指出每一步对应文件和数据结构。
- [ ] 能解释三个恢复模型为何不能互换，以及哪些部分是第三方工作。
- [ ] 能手写天气→模型映射和跳帧状态机。
- [ ] 能解释并检查每个模型的 shape、颜色顺序和数值范围。
- [ ] 能区分相对提升、绝对提升、无 GT proxy 与 GT-based mAP。
- [ ] 能解释 softmax 阈值 heuristic 为什么不等于真正多标签学习。
- [ ] 能解释论文最强结果，也能主动说出数据泄漏、合成天气和统计不足。
- [ ] 能重建 Snow100K 训练的 TransWeather 路径，并提供权重 checksum、训练配置和可重复 smoke test。
- [ ] 能在干净环境一条命令启动前端、API、worker、数据库和存储。
- [ ] 能在不依赖 AI 直接生成整段代码的情况下增加一个 API 字段、一个数据库 migration 和一个指标单测。
- [ ] 能用中文和英文各完成 30 秒、2 分钟和 10 分钟版本讲述。

## 14. 最终定位

这个项目目前最适合被定位为：

> 一个有完整研究叙事和可视化原型的 applied computer vision / ML systems 毕业设计，强项是任务驱动评估、异构模型编排、Snow100K 上的专用 TransWeather 训练和端到端集成；当前迁移快照的短板是可复现性、真实生产部署、异步任务/持久化，以及删除大文件后缺少完整的模型来源映射。

它已经足够成为实习简历的重要项目，但最有效的升级不是继续堆新模型，而是先完成三件事：从迁移资产恢复一条可验证的完整运行路径、让关键实验可复现、把长任务后端工程化。完成这三件事后，你能同时面向 CV/ML、Backend 和 Software Engineering 实习讲出不同侧重点。
