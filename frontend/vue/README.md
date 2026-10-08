# RoadClear Vue 前端

这是 RoadClear 道路监控视频增强系统的 Vue 3 前端界面。

## 功能特性

- **视频增强**: 上传视频并选择增强算法模型
- **模型选择**: 支持 Auto(自动)、AOD-Net、PreNet、Simple 等多种模型
- **任务评估**: 对增强前后的视频进行 YOLO 检测对比评估
- **实时日志**: 显示调度器运行日志

## 模型说明

| 模型 | 说明 |
|------|------|
| Auto (自动) | 系统自动识别天气并选择最佳模型 |
| AOD-Net | 适用于雾天去雾增强 |
| PreNet | 适用于雨天去雨增强 |
| Simple | 轻量级双边滤波，节省算力 |
| AOD-Net → PreNet | 先去雾再去雨，适用于雾雨混合天气 |
| PreNet → AOD-Net | 先去雨再去雾，适用于雨雾混合天气 |

## 快速开始

### 安装依赖

```bash
cd frontend/vue
npm install
```

### 启动开发服务器

```bash
npm run dev
```

前端将在 http://localhost:5173 启动。

### 构建生产版本

```bash
npm run build
```

构建产物将输出到 `dist` 目录。

## 注意事项

1. 确保后端服务已在 http://localhost:8000 运行
2. 首次使用需等待模型加载，可能需要一些时间
3. 视频处理是计算密集型任务，请耐心等待

## 技术栈

- Vue 3
- Element Plus
- Vite
