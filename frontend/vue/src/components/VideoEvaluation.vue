<template>
  <div class="video-evaluation">
    <el-alert v-if="!evaluationAvailable" :title="evaluationReason" type="warning" :closable="false" />
    <el-alert
      v-if="!currentVideoId"
      title="提示"
      type="info"
      description="请先在「视频增强」标签页中上传并处理视频，获取 Video ID 后再进行评估。"
      :closable="false"
      show-icon
    />

    <!-- Evaluation Input -->
    <el-card class="input-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Search /></el-icon> 运行评估</span>
        </div>
      </template>

      <el-form :inline="true" :model="evalForm" class="eval-form" size="large">
        <el-form-item label="视频 ID">
          <el-input
            v-model="evalForm.videoId"
            placeholder="请输入 Video ID"
            clearable
            style="width: 350px"
            size="large"
          >
            <template #append>
              <el-button @click="useCurrentVideoId" v-if="currentVideoId">
                使用当前视频
              </el-button>
            </template>
          </el-input>
        </el-form-item>
        <el-form-item>
          <el-button
            type="primary"
            size="large"
            :loading="evaluating"
            :disabled="!evalForm.videoId || !evaluationAvailable"
            @click="startEvaluation"
          >
            <el-icon v-if="!evaluating"><Cpu /></el-icon>
            {{ evaluating ? '评估中...' : '开始评估' }}
          </el-button>
        </el-form-item>
      </el-form>
    </el-card>

    <!-- Progress -->
    <el-card v-if="evaluating" class="progress-card" shadow="hover">
      <el-progress
        :percentage="evalProgress"
        :status="evalProgressStatus"
        :stroke-width="26"
        striped
        striped-flow
      />
      <div class="progress-text">{{ evalProgressText }}</div>
    </el-card>

    <!-- Evaluation Results -->
    <el-card v-if="showResults" class="results-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><DataAnalysis /></el-icon> 评估结果</span>
        </div>
      </template>

      <!-- Metrics Table -->
      <div class="metrics-section">
        <h3>检测指标对比</h3>
        <el-table :data="metricsTable" border stripe style="width: 100%" size="large">
          <el-table-column prop="indicator" label="指标" width="220" />
          <el-table-column prop="original" label="原始视频" />
          <el-table-column prop="enhanced" label="增强后视频" />
        </el-table>
        <div class="fps-note">{{ fpsNote }}</div>
      </div>

      <!-- Download Report -->
      <div class="download-section">
        <el-button type="success" size="large" @click="downloadReport" :loading="downloading">
          <el-icon><Download /></el-icon>
          下载评估报告 (JSON)
        </el-button>
      </div>

      <!-- HTML Report Preview -->
      <div class="html-report-section">
        <h3>HTML 报告预览</h3>
        <div class="report-preview" v-html="htmlReport"></div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, watch } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Search,
  Cpu,
  DataAnalysis,
  Download
} from '@element-plus/icons-vue'

const evaluationAvailable = ref(false)
const evaluationReason = ref('正在检查检测模型')
onMounted(async () => {
  try {
    const response = await fetch('/models/status')
    const models = (await response.json()).models
    evaluationAvailable.value = models.vehicle?.available && models.plate?.available
    evaluationReason.value = [models.vehicle, models.plate].filter(m => !m?.available).map(m => m?.reason || '模型不可用').join('; ')
  } catch { evaluationReason.value = '无法获取模型状态' }
})
const props = defineProps({
  backendUrl: {
    type: String,
    default: ''  // 使用相对路径，通过 Vite 代理
  },
  currentVideoId: {
    type: String,
    default: ''
  }
})

  // API 基础 URL - 直接使用空字符串，通过 Vite 代理访问后端
  const apiBase = ''

// Refs
const evalForm = ref({
  videoId: ''
})
const evaluating = ref(false)
const evalProgress = ref(0)
const evalProgressText = ref('等待评估...')
const showResults = ref(false)
const downloading = ref(false)

// Evaluation results
const evalResults = ref(null)
const htmlReport = ref('')
watch(() => props.currentVideoId, (id) => {
  if (id) {
    evalForm.value.videoId = id
    showResults.value = false
    evalResults.value = null
    htmlReport.value = ''
  }
}, { immediate: true })

// Computed
const evalProgressStatus = computed(() => {
  if (evalProgress.value === 100) return 'success'
  return undefined
})

const metricsTable = computed(() => {
  if (!evalResults.value) return []

  const metrics = evalResults.value.metrics || {}
  const before = metrics.before || {}
  const after = metrics.after || {}
  const imageQuality = metrics.image_quality || {}
  const performance = metrics.performance || {}

  const confDiff = (after.mean_confidence || 0) - (before.mean_confidence || 0)
  const confSign = (imageQuality.detection_count_change_rate || 0) >= 0 ? '+' : ''

  return [
    { indicator: '平均置信度', original: (before.mean_confidence || 0).toFixed(4), enhanced: (after.mean_confidence || 0).toFixed(4) },
    { indicator: '高置信度检测数 (≥0.7)', original: `${before.high_conf_count || 0}`, enhanced: `${after.high_conf_count || 0}` },
    { indicator: '检测车牌数', original: `${before.license_plate_count || 0}`, enhanced: `${after.license_plate_count || 0}` },
    { indicator: '检测数量变化率', original: `${before.total_detections || 0} (总数)`, enhanced: `${confSign}${((imageQuality.detection_count_change_rate || 0) * 100).toFixed(1)}%` },
    { indicator: '图像对比度增益', original: '-', enhanced: `${((imageQuality.contrast_gain || 0) * 100).toFixed(2)}%` },
    { indicator: '噪声抑制率', original: '-', enhanced: `${((imageQuality.noise_suppression_rate || 0) * 100).toFixed(2)}%` },
    { indicator: '时序稳定性', original: (before.temporal_stability || 0).toFixed(4), enhanced: (after.temporal_stability || 0).toFixed(4) },
    { indicator: '评估处理 FPS *', original: performance.base_fps?.toFixed(2) || '-', enhanced: performance.enhanced_fps?.toFixed(2) || '-' }
  ]
})

// 添加 FPS 说明
const fpsNote = computed(() => {
  return '* 评估处理 FPS：指评估系统处理视频的速度（包含检测、增强等操作），而非原始视频帧率。'
})

// Methods
const useCurrentVideoId = () => {
  if (props.currentVideoId) {
    evalForm.value.videoId = props.currentVideoId
  }
}

const startEvaluation = async () => {
  if (!evalForm.value.videoId) {
    ElMessage.warning('请输入 Video ID')
    return
  }

  evaluating.value = true
  showResults.value = false
  evalProgress.value = 20
  evalProgressText.value = '正在调用评估接口...'

  try {
    evalProgress.value = 40

    const response = await fetch(
      `${apiBase}/eval/evaluate/${evalForm.value.videoId}`,
      {
        method: 'POST'
      }
    )

    if (!response.ok) {
      throw new Error((await response.json()).detail || '评估失败')
    }

    evalProgress.value = 80
    evalProgressText.value = '正在整理评估结果...'

    const data = await response.json()

    evalResults.value = data
    htmlReport.value = data.html_report || ''

    evalProgress.value = 100
    evalProgressText.value = '评估完成!'
    showResults.value = true

    ElMessage.success('评估完成!')

  } catch (error) {
    ElMessage.error('评估失败: ' + error.message)
    evalProgress.value = 0
    evalProgressText.value = '评估失败'
  } finally {
    evaluating.value = false
  }
}

const downloadReport = async () => {
  if (!evalResults.value) return

  downloading.value = true

  try {
    const jsonStr = JSON.stringify(evalResults.value, null, 2)
    const blob = new Blob([jsonStr], { type: 'application/json' })
    const url = URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = url
    a.download = `evaluation_${evalForm.value.videoId}.json`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)

    ElMessage.success('报告下载成功!')
  } catch (error) {
    ElMessage.error('下载失败: ' + error.message)
  } finally {
    downloading.value = false
  }
}
</script>

<style scoped>
.video-evaluation {
  display: flex;
  flex-direction: column;
  gap: 24px;
  font-size: 18px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
  font-size: 20px;
}

.card-header span {
  display: flex;
  align-items: center;
  gap: 10px;
}

/* Progress */
.progress-card {
  text-align: center;
}

.progress-text {
  margin-top: 16px;
  font-size: 20px;
  color: #606266;
  font-weight: 500;
}

/* Results */
.metrics-section {
  margin-bottom: 28px;
}

.metrics-section h3 {
  font-size: 22px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 20px;
}

.fps-note {
  margin-top: 16px;
  font-size: 16px;
  color: #909399;
  font-style: italic;
  line-height: 1.6;
}

.download-section {
  margin: 28px 0;
  text-align: center;
}

.html-report-section {
  margin-top: 28px;
}

.html-report-section h3 {
  font-size: 22px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 20px;
}

.report-preview {
  background: #f5f7fa;
  border-radius: 10px;
  padding: 20px;
  max-height: 600px;
  overflow: auto;
}

.report-preview :deep(table) {
  width: 100%;
  border-collapse: collapse;
}

.report-preview :deep(th),
.report-preview :deep(td) {
  border: 1px solid #dcdfe6;
  padding: 12px 16px;
  text-align: center;
  font-size: 16px;
}

.report-preview :deep(th) {
  background: #f5f7fa;
  font-weight: 700;
  font-size: 17px;
  color: #303133;
}

.report-preview :deep(img) {
  max-width: 100%;
  height: auto;
}

/* 表单标签字号 */
:deep(.el-form-item__label) {
  font-size: 18px;
  font-weight: 500;
}

/* Element Plus 表格字号覆盖 */
:deep(.el-table) {
  font-size: 17px;
}

:deep(.el-table th) {
  font-size: 18px;
  font-weight: 600;
}

:deep(.el-table td) {
  font-size: 17px;
}

/* Alert 提示字号 */
:deep(.el-alert__title) {
  font-size: 18px;
}

:deep(.el-alert__description) {
  font-size: 16px;
}
</style>
