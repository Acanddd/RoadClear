<template>
  <!-- Historical images and confidence proxies, not labeled benchmark metrics. -->
  <el-card class="showcase-card yolo-card" shadow="hover">
    <template #header>
      <div class="card-header">
        <span><el-icon><View /></el-icon> YOLO检测增强效果对比</span>
        <el-tag type="primary" size="small">目标检测</el-tag>
      </div>
    </template>

    <div class="yolo-comparison-section">
      <el-alert title="历史对比图与置信度代理分数：不是当前视频的标注评测，不能作为 mAP、Precision 或 Recall。" type="info" :closable="false" />
      <!-- 对比图区域 -->
      <div class="comparison-grid">
        <!-- 原始图检测 -->
        <div class="detection-panel">
          <div class="panel-header">
            <h4>下雪场景检测</h4>
            <el-tag type="danger" size="small">去雪前</el-tag>
          </div>
          <div class="canvas-container">
            <canvas ref="originalCanvas" class="detection-canvas"></canvas>
            <div v-if="loading" class="loading-overlay">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>正在检测...</span>
            </div>
          </div>
          <div class="detection-stats">
            <div class="stat-item">
              <span class="stat-label">检测数量:</span>
              <span class="stat-value">{{ originalDetections.length }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">平均置信度:</span>
              <span class="stat-value">{{ originalAvgConf }}%</span>
            </div>
          </div>
        </div>

        <!-- 中间指标对比 -->
        <div class="metrics-panel">
          <div class="metrics-header">
            <el-icon><TrendCharts /></el-icon>
            <h4>量化指标对比</h4>
          </div>

          <el-alert v-if="!metricsReady" :title="metricsError || '正在加载检测结果'" type="info" :closable="false" />
          <div v-if="metricsReady" class="metrics-content">
            <!-- mAP 对比 -->
            <div class="metric-item">
              <div class="metric-label">
                <el-icon><Histogram /></el-icon>
                <span>置信度代理分数</span>
              </div>
              <div class="metric-comparison">
                <div class="metric-bar-container">
                  <div class="metric-bar original-bar" :style="{ width: originalMAP + '%' }">
                    <span class="bar-label">{{ originalMAP }}%</span>
                  </div>
                </div>
                <div class="metric-bar-container">
                  <div class="metric-bar enhanced-bar" :style="{ width: enhancedMAP + '%' }">
                    <span class="bar-label">{{ enhancedMAP }}%</span>
                  </div>
                </div>
              </div>
              <div class="metric-improvement">
                <el-icon><Top /></el-icon>
                提升: <strong>+{{ (enhancedMAP - originalMAP).toFixed(1) }}%</strong>
              </div>
            </div>

            <!-- 置信度代理 P 对比 -->
            <div class="metric-item">
              <div class="metric-label">
                <el-icon><CircleCheck /></el-icon>
                <span>置信度代理 P</span>
              </div>
              <div class="metric-comparison">
                <div class="metric-bar-container">
                  <div class="metric-bar original-bar" :style="{ width: originalPrecision + '%' }">
                    <span class="bar-label">{{ originalPrecision }}%</span>
                  </div>
                </div>
                <div class="metric-bar-container">
                  <div class="metric-bar enhanced-bar" :style="{ width: enhancedPrecision + '%' }">
                    <span class="bar-label">{{ enhancedPrecision }}%</span>
                  </div>
                </div>
              </div>
              <div class="metric-improvement">
                <el-icon><Top /></el-icon>
                提升: <strong>+{{ (enhancedPrecision - originalPrecision).toFixed(1) }}%</strong>
              </div>
            </div>

            <!-- 高置信度占比代理 R 对比 -->
            <div class="metric-item">
              <div class="metric-label">
                <el-icon><Aim /></el-icon>
                <span>高置信度占比代理 R</span>
              </div>
              <div class="metric-comparison">
                <div class="metric-bar-container">
                  <div class="metric-bar original-bar" :style="{ width: originalRecall + '%' }">
                    <span class="bar-label">{{ originalRecall }}%</span>
                  </div>
                </div>
                <div class="metric-bar-container">
                  <div class="metric-bar enhanced-bar" :style="{ width: enhancedRecall + '%' }">
                    <span class="bar-label">{{ enhancedRecall }}%</span>
                  </div>
                </div>
              </div>
              <div class="metric-improvement">
                <el-icon><Top /></el-icon>
                提升: <strong>+{{ (enhancedRecall - originalRecall).toFixed(1) }}%</strong>
              </div>
            </div>

            <!-- F1 Score 对比 -->
            <div class="metric-item">
              <div class="metric-label">
                <el-icon><DataLine /></el-icon>
                <span>F1 Score</span>
              </div>
              <div class="metric-comparison">
                <div class="metric-bar-container">
                  <div class="metric-bar original-bar" :style="{ width: originalF1 + '%' }">
                    <span class="bar-label">{{ originalF1 }}%</span>
                  </div>
                </div>
                <div class="metric-bar-container">
                  <div class="metric-bar enhanced-bar" :style="{ width: enhancedF1 + '%' }">
                    <span class="bar-label">{{ enhancedF1 }}%</span>
                  </div>
                </div>
              </div>
              <div class="metric-improvement">
                <el-icon><Top /></el-icon>
                提升: <strong>+{{ (enhancedF1 - originalF1).toFixed(1) }}%</strong>
              </div>
            </div>
          </div>

          <el-divider />

          <!-- 类别统计 -->
          <div class="class-stats">
            <h5>各类别检测统计</h5>
            <div class="class-grid">
              <div v-for="cls in classStats" :key="cls.name" class="class-item">
                <div class="class-name">{{ cls.name }}</div>
                <div class="class-count">
                  <span class="count-before">{{ cls.before }}</span>
                  <el-icon><Right /></el-icon>
                  <span class="count-after">{{ cls.after }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- 增强图检测 -->
        <div class="detection-panel">
          <div class="panel-header">
            <h4>去雪后检测</h4>
            <el-tag type="success" size="small">去雪后</el-tag>
          </div>
          <div class="canvas-container">
            <canvas ref="enhancedCanvas" class="detection-canvas"></canvas>
            <div v-if="loading" class="loading-overlay">
              <el-icon class="is-loading"><Loading /></el-icon>
              <span>正在检测...</span>
            </div>
          </div>
          <div class="detection-stats">
            <div class="stat-item">
              <span class="stat-label">检测数量:</span>
              <span class="stat-value success">{{ enhancedDetections.length }}</span>
            </div>
            <div class="stat-item">
              <span class="stat-label">平均置信度:</span>
              <span class="stat-value success">{{ enhancedAvgConf }}%</span>
            </div>
          </div>
        </div>
      </div>

      <el-divider />

      <!-- 说明文字 -->
      <div class="description-text">
        <h4>YOLO目标检测增强效果分析</h4>
        <p>
          使用 YOLOv5s (DETRAC 4-class) 模型对监控视角的雪天场景进行车辆目标检测对比。
          左侧为下雪场景的检测结果，右侧为经过 TransWeather 去雪后的检测结果。
          本页用于查看历史样例，解读时请区分以下信息：
        </p>
        <ul class="benefit-list">
          <li><el-icon><Check /></el-icon> 检测框数量变化不等于漏检率变化</li>
          <li><el-icon><Check /></el-icon> 检测置信度变化不等于误检率变化</li>
          <li><el-icon><Check /></el-icon> 历史对比图；无标注代理分数不能证明检测准确率提升</li>
          <li><el-icon><Check /></el-icon> 小目标效果需要带标注数据另行验证</li>
        </ul>
        <p class="tech-note">
          <el-icon><InfoFilled /></el-icon>
          检测模型: YOLOv5s (DETRAC 4-class) | 视角: 监控俯视 | 去雪: TransWeather | 置信度阈值: 0.1
        </p>
      </div>
    </div>
  </el-card>
</template>

<script setup>
import { ref, onMounted, computed } from 'vue'
import {
  View,
  Loading,
  TrendCharts,
  Histogram,
  CircleCheck,
  Aim,
  DataLine,
  Top,
  Right,
  Check,
  InfoFilled
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

// Canvas 引用
const originalCanvas = ref(null)
const enhancedCanvas = ref(null)
const loading = ref(true)
const metricsReady = ref(false)
const metricsError = ref('')

// 检测结果
const originalDetections = ref([])
const enhancedDetections = ref([])

// 类别映射 (DETRAC 4-class)
const classNames = ['car', 'bus', 'van', 'others']
const classColors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

// 模拟的量化指标
const originalMAP = ref(0)
const enhancedMAP = ref(0)
const originalPrecision = ref(0)
const enhancedPrecision = ref(0)
const originalRecall = ref(0)
const enhancedRecall = ref(0)
const originalF1 = ref(0)
const enhancedF1 = ref(0)

// 计算平均置信度
const originalAvgConf = computed(() => {
  if (originalDetections.value.length === 0) return '0.0'
  const avg = originalDetections.value.reduce((sum, det) => sum + det.confidence, 0) / originalDetections.value.length
  return (avg * 100).toFixed(1)
})

const enhancedAvgConf = computed(() => {
  if (enhancedDetections.value.length === 0) return '0.0'
  const avg = enhancedDetections.value.reduce((sum, det) => sum + det.confidence, 0) / enhancedDetections.value.length
  return (avg * 100).toFixed(1)
})

// 类别统计
const classStats = computed(() => {
  return classNames.map((name, idx) => ({
    name: name.toUpperCase(),
    before: originalDetections.value.filter(d => d.class === idx).length,
    after: enhancedDetections.value.filter(d => d.class === idx).length
  }))
})

// 加载图片并绘制检测框
const loadAndDrawDetections = async () => {
  loading.value = true

  try {
    // 调用后端真实检测API
    const response = await fetch('/detection/detect/showcase', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    })

    if (!response.ok) {
      throw new Error(`检测API调用失败: ${response.status}`)
    }

    const result = await response.json()

    if (!result.success) {
      throw new Error('检测失败')
    }

    const data = result.data

    // 加载图片
    const originalImg = await loadImage('/algorithm-showcase/yolo-snowy-detected.png')
    const enhancedImg = await loadImage('/algorithm-showcase/yolo-enhanced-detected.png')

    // 转换检测结果格式
    originalDetections.value = data.original.detections.map(det => ({
      class: det.class,
      x: det.bbox.center_x / data.original.image_size.width,
      y: det.bbox.center_y / data.original.image_size.height,
      w: det.bbox.width / data.original.image_size.width,
      h: det.bbox.height / data.original.image_size.height,
      confidence: det.confidence
    }))

    enhancedDetections.value = data.enhanced.detections.map(det => ({
      class: det.class,
      x: det.bbox.center_x / data.enhanced.image_size.width,
      y: det.bbox.center_y / data.enhanced.image_size.height,
      w: det.bbox.width / data.enhanced.image_size.width,
      h: det.bbox.height / data.enhanced.image_size.height,
      confidence: det.confidence
    }))

    metricsReady.value = true
    // 更新置信度代理指标
    originalMAP.value = data.original.metrics.mAP
    enhancedMAP.value = data.enhanced.metrics.mAP
    originalPrecision.value = data.original.metrics.precision
    enhancedPrecision.value = data.enhanced.metrics.precision
    originalRecall.value = data.original.metrics.recall
    enhancedRecall.value = data.enhanced.metrics.recall
    originalF1.value = data.original.metrics.f1_score
    enhancedF1.value = data.enhanced.metrics.f1_score

    // 绘制检测结果
    drawDetections(originalCanvas.value, originalImg, originalDetections.value)
    drawDetections(enhancedCanvas.value, enhancedImg, enhancedDetections.value)

    loading.value = false
    ElMessage.success('检测完成')
  } catch (error) {
    metricsReady.value = false
    metricsError.value = error.message
    console.error('加载检测数据失败:', error)
    ElMessage.error(`加载检测数据失败: ${error.message}`)
    loading.value = false
  }
}

// 加载图片
const loadImage = (src) => {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}



// 绘制检测框
const drawDetections = (canvas, image, detections) => {
  if (!canvas || !image) return

  const ctx = canvas.getContext('2d')
  canvas.width = image.width
  canvas.height = image.height
  ctx.drawImage(image, 0, 0)

  detections.forEach(det => {
    const x = (det.x - det.w / 2) * image.width
    const y = (det.y - det.h / 2) * image.height
    const w = det.w * image.width
    const h = det.h * image.height
    const color = classColors[det.class]

    ctx.strokeStyle = color
    ctx.lineWidth = 3
    ctx.strokeRect(x, y, w, h)

    const label = `${classNames[det.class]} ${(det.confidence * 100).toFixed(1)}%`
    ctx.font = 'bold 14px Arial'
    const textWidth = ctx.measureText(label).width

    ctx.fillStyle = color
    ctx.fillRect(x, y - 25, textWidth + 10, 25)
    ctx.fillStyle = '#fff'
    ctx.fillText(label, x + 5, y - 7)
  })
}

onMounted(() => {
  loadAndDrawDetections()
})
</script>

<style scoped>
.yolo-card {
  border-radius: 12px;
  overflow: hidden;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 16px;
}

.card-header span {
  display: flex;
  align-items: center;
  gap: 8px;
}

.yolo-comparison-section {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.comparison-grid {
  display: grid;
  grid-template-columns: 1fr 400px 1fr;
  gap: 20px;
  align-items: start;
}

.detection-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.panel-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px 12px;
  background: #f5f7fa;
  border-radius: 6px;
}

.panel-header h4 {
  margin: 0;
  font-size: 15px;
  font-weight: 600;
  color: #303133;
}

.canvas-container {
  position: relative;
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background: #000;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.detection-canvas {
  width: 100%;
  height: auto;
  display: block;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  color: #fff;
  font-size: 14px;
}

.detection-stats {
  display: flex;
  justify-content: space-around;
  padding: 12px;
  background: #f5f7fa;
  border-radius: 6px;
}

.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
}

.stat-label {
  font-size: 12px;
  color: #909399;
}

.stat-value {
  font-size: 18px;
  font-weight: 700;
  color: #303133;
}

.stat-value.success {
  color: #67c23a;
}

.metrics-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
  padding: 16px;
  background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
  border-radius: 8px;
  border: 2px solid #667eea;
}

.metrics-header {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #667eea;
}

.metrics-header h4 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
}

.metrics-content {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.metric-item {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.metric-label {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.metric-comparison {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.metric-bar-container {
  position: relative;
  width: 100%;
  height: 28px;
  background: #f5f7fa;
  border-radius: 4px;
  overflow: hidden;
}

.metric-bar {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
  transition: width 1s ease;
  border-radius: 4px;
}

.original-bar {
  background: linear-gradient(90deg, #f56c6c, #ff8787);
}

.enhanced-bar {
  background: linear-gradient(90deg, #67c23a, #85ce61);
}

.bar-label {
  font-size: 12px;
  font-weight: 700;
  color: #fff;
  text-shadow: 0 1px 2px rgba(0, 0, 0, 0.3);
}

.metric-improvement {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 13px;
  color: #67c23a;
  padding: 4px 8px;
  background: #f0f9ff;
  border-radius: 4px;
  border-left: 3px solid #67c23a;
}

.metric-improvement strong {
  font-size: 14px;
}

.class-stats {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.class-stats h5 {
  margin: 0;
  font-size: 14px;
  font-weight: 600;
  color: #303133;
}

.class-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 8px;
}

.class-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 8px;
  background: #fff;
  border-radius: 4px;
  border: 1px solid #dcdfe6;
}

.class-name {
  font-size: 12px;
  font-weight: 600;
  color: #606266;
  text-transform: uppercase;
}

.class-count {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 14px;
}

.count-before {
  color: #f56c6c;
  font-weight: 600;
}

.count-after {
  color: #67c23a;
  font-weight: 700;
}

.description-text {
  padding: 12px;
}

.description-text h4 {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
  margin: 0 0 12px 0;
}

.description-text p {
  font-size: 14px;
  line-height: 1.8;
  color: #606266;
  margin: 0 0 12px 0;
  text-align: justify;
}

.benefit-list {
  list-style: none;
  padding: 0;
  margin: 12px 0;
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.benefit-list li {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 14px;
  color: #606266;
  padding: 8px 12px;
  background: #f0f9ff;
  border-radius: 4px;
  border-left: 3px solid #409eff;
}

.benefit-list li .el-icon {
  color: #67c23a;
  font-size: 16px;
}

.tech-note {
  display: flex;
  align-items: center;
  gap: 8px;
  background: #ecf5ff;
  padding: 12px;
  border-radius: 6px;
  border-left: 4px solid #409eff;
  font-size: 13px;
  color: #409eff;
  margin-top: 12px;
}

@media (max-width: 1400px) {
  .comparison-grid {
    grid-template-columns: 1fr;
    gap: 24px;
  }

  .metrics-panel {
    order: 3;
  }
}
</style>
