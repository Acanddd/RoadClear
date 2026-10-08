<template>
  <div class="algorithm-showcase">
    <!-- Grad-CAM 热力图 -->
    <el-card class="showcase-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><PictureFilled /></el-icon> Grad-CAM 热力图</span>
          <el-tag type="warning" size="small">注意力可视化</el-tag>
        </div>
      </template>
      <div class="gradcam-section">
        <div class="image-container">
          <img
            :src="gradcamImage"
            alt="Grad-CAM 热力图"
            class="showcase-image"
            @error="handleImageError('gradcam')"
          />
        </div>
        <div class="description-text">
          <h4>Grad-CAM 可视化说明</h4>
          <p>
            Grad-CAM（Gradient-weighted Class Activation Mapping）通过梯度加权类激活映射技术，
            将深度学习模型的注意力区域以热力图形式呈现。红色区域表示模型关注度最高的部分，
            蓝色区域表示关注度较低。这有助于理解模型在处理恶劣天气视频时重点关注的图像特征。
          </p>
          <el-divider />
          <p class="tech-note">
            <el-icon><InfoFilled /></el-icon>
            热力图颜色越暖（红-黄），表示该区域对模型决策的贡献越大
          </p>
        </div>
      </div>
    </el-card>

    <!-- 像素残差图 -->
    <el-card class="showcase-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Grid /></el-icon> 像素残差对比图</span>
          <el-tag type="info" size="small">残差分析</el-tag>
        </div>
      </template>
      <div class="residual-section">
        <div class="residual-grid">
          <div class="residual-item">
            <div class="residual-label">原始退化图</div>
            <div class="image-container">
              <img
                :src="residualOriginal"
                alt="原始图"
                class="showcase-image"
                @error="handleImageError('residual-original')"
              />
            </div>
            <p class="image-caption">包含天气退化的原始输入</p>
          </div>

          <div class="residual-item">
            <div class="residual-label residual-highlight">剥离的残差图</div>
            <div class="image-container">
              <img
                :src="residualDiff"
                alt="残差图"
                class="showcase-image"
                @error="handleImageError('residual-diff')"
              />
            </div>
            <p class="image-caption">网络提取的退化成分</p>
          </div>

          <div class="residual-item">
            <div class="residual-label">增强后清晰图</div>
            <div class="image-container">
              <img
                :src="residualEnhanced"
                alt="增强图"
                class="showcase-image"
                @error="handleImageError('residual-enhanced')"
              />
            </div>
            <p class="image-caption">去除退化后的清晰结果</p>
          </div>
        </div>
        <el-divider />
        <div class="description-text">
          <h4>残差学习原理</h4>
          <p>
            残差图展示了网络从原始图像中分离出的天气退化成分（如雨滴、雾霾、雪花等）。
            通过学习残差映射 R(x)，使得 <code>增强图 = 原始图 - 残差图</code>，
            从而实现对退化因素的精准去除。中间的残差图越清晰，说明网络对退化特征的提取越准确。
          </p>
        </div>
      </div>
    </el-card>

    <!-- 网络特征图 -->
    <el-card class="showcase-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><DataAnalysis /></el-icon> 网络特征图</span>
          <el-tag type="success" size="small">深层特征</el-tag>
        </div>
      </template>
      <div class="feature-section">
        <div class="image-container">
          <img
            :src="featureMap"
            alt="网络特征图"
            class="showcase-image"
            @error="handleImageError('feature')"
          />
        </div>
        <div class="description-text">
          <h4>卷积神经网络特征可视化</h4>
          <p>
            特征图展示了卷积神经网络中间层提取的抽象特征表示。浅层特征图捕捉边缘、纹理等低级视觉信息，
            深层特征图则学习到语义级别的高级特征（如车辆轮廓、道路结构等）。
            通过可视化特征图，可以直观理解网络如何逐层抽象和处理视觉信息。
          </p>
          <el-divider />
          <p class="tech-note">
            <el-icon><InfoFilled /></el-icon>
            不同通道的特征图对应不同的视觉模式检测器
          </p>
        </div>
      </div>
    </el-card>

    <!-- YOLO检测增强效果对比 -->
    <YoloComparisonCard />

    <!-- 天气退化网络调度工作流 -->
    <el-card class="showcase-card workflow-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Connection /></el-icon> 天气退化网络调度工作流</span>
          <el-tag type="danger" size="small">实时动画</el-tag>
        </div>
      </template>
      <div class="workflow-section">
        <div class="workflow-container">
          <iframe
            :src="workflowUrl"
            frameborder="0"
            class="workflow-iframe"
            title="天气退化网络调度工作流动画"
          ></iframe>
        </div>
        <el-divider />
        <div class="description-text">
          <h4>自适应网络调度机制</h4>
          <p>
            系统采用智能天气分类器对输入视频进行实时分析，根据检测到的天气类型（雨、雾、雪等）
            动态调度相应的专用去退化子网络。工作流展示了从输入到输出的完整处理链路，
            包括天气识别、分支选择、特征融合和视频增强等关键步骤。
          </p>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed } from 'vue'
import {
  PictureFilled,
  Grid,
  DataAnalysis,
  Connection,
  InfoFilled
} from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import YoloComparisonCard from './YoloComparisonCard.vue'

// 图片资源路径（使用 public 目录）
const gradcamImage = ref('/algorithm-showcase/gradcam.png')
const residualOriginal = ref('/algorithm-showcase/residual-original.png')
const residualDiff = ref('/algorithm-showcase/residual-diff.png')
const residualEnhanced = ref('/algorithm-showcase/residual-enhanced.png')
const featureMap = ref('/algorithm-showcase/feature-map.png')

// Workflow HTML 路径
const workflowUrl = computed(() => '/workflow.html')

// 图片加载错误处理
const handleImageError = (type) => {
  console.warn(`图片加载失败: ${type}`)
  // 可以设置默认占位图
}
</script>

<style scoped>
.algorithm-showcase {
  display: flex;
  flex-direction: column;
  gap: 24px;
  padding: 4px;
}

.showcase-card {
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

/* Grad-CAM Section */
.gradcam-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: start;
}

.image-container {
  width: 100%;
  border-radius: 8px;
  overflow: hidden;
  background: #f5f7fa;
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 300px;
}

.showcase-image {
  width: 100%;
  height: auto;
  display: block;
  object-fit: contain;
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

.description-text code {
  background: #f5f7fa;
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Consolas', 'Monaco', monospace;
  color: #e6a23c;
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
}

/* Residual Section */
.residual-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.residual-grid {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 20px;
}

.residual-item {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.residual-label {
  font-size: 15px;
  font-weight: 600;
  color: #303133;
  text-align: center;
  padding: 8px;
  background: #f5f7fa;
  border-radius: 6px;
}

.residual-highlight {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.image-caption {
  font-size: 13px;
  color: #909399;
  text-align: center;
  margin: 0;
}

/* Feature Section */
.feature-section {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  align-items: start;
}

/* Workflow Section */
.workflow-card {
  min-height: 600px;
}

.workflow-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.workflow-container {
  width: 100%;
  height: 720px;
  border-radius: 8px;
  overflow: hidden;
  background: #06090f;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.workflow-iframe {
  width: 100%;
  height: 100%;
  border: none;
}

/* Responsive Design */
@media (max-width: 1200px) {
  .gradcam-section,
  .feature-section {
    grid-template-columns: 1fr;
  }

  .residual-grid {
    grid-template-columns: 1fr;
  }
}

@media (max-width: 768px) {
  .workflow-container {
    height: 500px;
  }
}
</style>
