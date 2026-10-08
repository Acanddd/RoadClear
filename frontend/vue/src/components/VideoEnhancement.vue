<template>
  <div class="video-enhancement">
    <el-alert title="本机演示：单任务；最大 100 MB、300 帧、最长边 1920px，总解码像素不超过 1.5 亿。请使用短视频。" type="info" :closable="false" />
    <!-- Model Selection Panel -->
    <el-card class="model-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Setting /></el-icon> 算法模型选择</span>
        </div>
      </template>
      <el-select v-model="selectedModel" placeholder="请选择增强模型" style="width: 100%" size="large">
        <el-option
          v-for="item in modelOptions"
          :key="item.value"
          :label="item.label"
          :value="item.value"
          :disabled="!modelAvailable(item.value)"
        >
          <span class="model-option-label">{{ item.label }}</span>
          <span class="model-option-desc">{{ modelAvailable(item.value) ? item.desc : unavailableReason(item.value) }}</span>
        </el-option>
      </el-select>
      <div class="model-desc-text">
        <el-tag :type="modelTagType" size="small">{{ currentModelDesc }}</el-tag>
      </div>
    </el-card>

    <!-- Upload Section -->
    <el-card class="upload-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Upload /></el-icon> 上传视频</span>
        </div>
      </template>
      <el-upload
        ref="uploadRef"
        class="video-uploader"
        drag
        :auto-upload="false"
        :on-change="handleFileChange"
        :on-error="handleUploadError"
        :before-upload="beforeUpload"
        accept="video/*"
        :limit="1"
      >
        <el-icon class="upload-icon"><UploadFilled /></el-icon>
        <div class="upload-text">
          拖拽视频到此处 或 <em>点击上传</em>
        </div>
        <template #tip>
          <div class="upload-tip">
            支持 MP4、AVI 等常见视频格式
          </div>
        </template>
      </el-upload>

      <div class="upload-actions">
        <el-button
          type="primary"
          size="large"
          :loading="processing"
          :disabled="!videoFile"
          @click="handleProcess"
        >
          <el-icon v-if="!processing"><VideoPlay /></el-icon>
          {{ processing ? '处理中...' : '开始增强' }}
        </el-button>
        <el-button size="large" @click="resetUpload" :disabled="processing">
          <el-icon><RefreshLeft /></el-icon>
          重置
        </el-button>
      </div>
      <div class="post-process-toggle">
        <el-switch
          v-model="postProcessEnabled"
          active-text="启用增强后处理"
          inactive-text="关闭增强后处理"
          size="large"
        />
        <span class="post-process-note">开启后，增强结果将执行 CLAHE、Unsharp Mask、降噪与高光动态范围压缩。</span>
      </div>

      <!-- Multi-label Weather Recognition -->
      <div class="multi-label-section">
        <el-divider content-position="left">
          <el-icon><Setting /></el-icon>
          <span class="multi-label-title">多标签天气识别</span>
        </el-divider>
        <el-alert
          title="多标签模式说明"
          type="info"
          :closable="false"
          show-icon
          style="margin-bottom: 16px"
        >
          <template #default>
            <p style="margin: 0 0 8px 0">启用多标签模式后，系统可以同时识别多种天气（如雨天+雾天），并级联使用多个模型进行处理。</p>
            <p style="margin: 0">例如：检测到 Rain(60%) + Fog(40%) 时，会先使用 PreNet 去雨，再使用 AODNet 去雾。</p>
          </template>
        </el-alert>
        <el-form label-width="180px" size="large">
          <el-form-item label="启用多标签识别">
            <el-switch
              v-model="multiLabelEnabled"
              active-text="开启"
              inactive-text="关闭"
              size="large"
            />
            <span class="param-desc">识别混合天气并级联处理（推荐开启）</span>
          </el-form-item>
          <el-form-item
            label="多标签概率阈值"
            v-if="multiLabelEnabled"
          >
            <el-slider
              v-model="multiLabelThreshold"
              :min="0.1"
              :max="0.5"
              :step="0.05"
              :marks="{
                0.2: '宽松',
                0.3: '推荐',
                0.4: '严格'
              }"
              show-input
              :format-tooltip="formatThresholdTooltip"
              style="width: 350px"
            />
            <span class="param-desc">概率超过此值的天气类别都会被处理</span>
          </el-form-item>
        </el-form>
      </div>
    </el-card>

    <!-- Progress Section -->
    <el-card v-if="processing || progress > 0" class="progress-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Loading /></el-icon> 处理进度</span>
        </div>
      </template>
      <el-progress
        :percentage="progress"
        :status="progressStatus"
        :stroke-width="26"
        striped
        striped-flow
      />
      <div class="progress-text">{{ progressText }}</div>
    </el-card>

    <!-- Video Comparison Section -->
    <el-card v-if="videoId" class="result-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><VideoCamera /></el-icon> 处理结果</span>
          <el-tag type="success">Video ID: {{ videoId }}</el-tag>
        </div>
      </template>

      <div class="video-comparison">
        <div class="video-panel">
          <h3>原始视频</h3>
          <video
            v-if="originalVideoUrl"
            :src="originalVideoUrl"
            controls
            playsinline
            class="video-player"
            @loadedmetadata="() => console.log('Original video metadata loaded')"
            @error="(e) => console.error('Original video error:', e.target.error)"
            @canplay="() => console.log('Original video can play')"
          />
          <div v-else class="video-placeholder">
            <el-icon :size="48"><VideoCamera /></el-icon>
            <span>原始视频</span>
          </div>
        </div>

        <div class="video-panel">
          <h3>增强后视频</h3>
          <video
            v-if="enhancedVideoUrl"
            :src="enhancedVideoUrl"
            controls
            playsinline
            class="video-player"
            @error="handleVideoError('enhanced')"
          />
          <div v-else class="video-placeholder">
            <el-icon :size="48"><VideoPlay /></el-icon>
            <span>增强后视频</span>
          </div>
          <p v-if="enhancedVideoError" style="color: red; font-size: 12px;">{{ enhancedVideoError }}</p>
        </div>
      </div>

      <!-- Model Info -->
      <el-divider />
      <div class="model-info">
        <el-descriptions :column="2" border size="large">
          <el-descriptions-item label="使用模型">
            <el-tag :type="modelTagType" size="large">{{ modelChain }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item label="模型说明">
            {{ modelDescription }}
          </el-descriptions-item>
        </el-descriptions>
      </div>
    </el-card>

    <!-- Logs Section -->
    <el-card v-if="videoId" class="logs-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Document /></el-icon> 调度日志</span>
          <el-button @click="fetchLogs" :loading="logsLoading">
            <el-icon><Refresh /></el-icon> 刷新
          </el-button>
        </div>
      </template>
      <div class="logs-container">
        <el-scrollbar height="300px">
          <pre class="logs-content">{{ logs || '暂无日志' }}</pre>
        </el-scrollbar>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import {
  Setting,
  Upload,
  UploadFilled,
  VideoPlay,
  RefreshLeft,
  Loading,
  VideoCamera,
  Document,
  Refresh
} from '@element-plus/icons-vue'

const props = defineProps({
  backendUrl: {
    type: String,
    default: ''  // 使用相对路径，通过 Vite 代理
  }
})

const emit = defineEmits(['video-processed'])

  // API 基础 URL - 直接使用空字符串，通过 Vite 代理访问后端
  const apiBase = ''

// Model options for dropdown
const capabilities = ref({})
const modelAvailable = (name) => name.split('->').every(n => capabilities.value[n]?.available === true)
const unavailableReason = (name) => name.split('->').filter(n => !capabilities.value[n]?.available).map(n => capabilities.value[n]?.reason || '正在检查模型状态').join('; ')
const refreshModels = async () => {
  try {
    const response = await fetch('/models/status')
    if (!response.ok) throw new Error('模型状态读取失败')
    capabilities.value = (await response.json()).models
  } catch { capabilities.value = {} }
}
onMounted(refreshModels)
onUnmounted(() => stopStatusPolling())
const modelOptions = [
  { value: 'auto', label: 'Auto (自动)', desc: '系统自动识别天气并选择最佳模型' },
  { value: 'aodnet', label: 'AOD-Net', desc: '适用于雾天去雾增强' },
  { value: 'prenet', label: 'PreNet', desc: '适用于雨天去雨增强' },
  { value: 'transweather', label: 'TransWeather', desc: '适用于雪天去雪增强' },
  { value: 'simple', label: 'Simple', desc: '轻量级双边滤波，节省算力' },
  { value: 'aodnet->prenet', label: 'AOD-Net → PreNet', desc: '先去雾再去雨，适用于雾雨混合天气' },
  { value: 'prenet->aodnet', label: 'PreNet → AOD-Net', desc: '先再去雾，适用于雨雾混合天气' },
  { value: 'transweather->prenet', label: 'TransWeather → PreNet', desc: '先去雪再去雨，适用于雪雨混合天气' },
  { value: 'aodnet->transweather', label: 'AOD-Net → TransWeather', desc: '先去雾再去雪，适用于雾雪混合天气' }
]

// Refs
const uploadRef = ref(null)
const selectedModel = ref('auto')
const videoFile = ref(null)
const processing = ref(false)
const progress = ref(0)
const progressText = ref('等待处理...')
const videoId = ref('')
const originalVideoUrl = ref('')
const enhancedVideoUrl = ref('')
const enhancedVideoError = ref('')
const modelChain = ref('')
const postProcessEnabled = ref(false)
const multiLabelEnabled = ref(false)
const multiLabelThreshold = ref(0.3)
const logs = ref('')
const logsLoading = ref(false)
let statusTimer = null

// Computed
const uploadUrl = computed(() => `${apiBase}/video/upload_video`)

const progressStatus = computed(() => {
  if (progress.value === 100) return 'success'
  if (progress.value > 0) return undefined
  return 'exception'
})

const currentModelDesc = computed(() => {
  const model = modelOptions.find(m => m.value === selectedModel.value)
  return model ? model.desc : ''
})

const modelDescription = computed(() => currentModelDesc.value)

const modelTagType = computed(() => {
  const types = {
    'auto': 'success',
    'aodnet': 'warning',
    'prenet': 'info',
    'transweather': 'info',
    'simple': 'info',
    'aodnet->prenet': 'danger',
    'prenet->aodnet': 'danger',
    'transweather->prenet': 'danger',
    'aodnet->transweather': 'danger'
  }
  return types[selectedModel.value] || 'info'
})

const formatThresholdTooltip = (value) => {
  return `${(value * 100).toFixed(0)}%`
}

// Methods
const beforeUpload = (file) => {
  return false // Prevent auto upload
}

const handleFileChange = (file) => {
  // Element Plus upload 组件的 onChange 返回包装对象，raw 是原始 File
  videoFile.value = file.raw
  console.log('File selected:', file.raw)
  console.log('File name:', file.raw?.name)
  console.log('File size:', file.raw?.size)
  console.log('File type:', file.raw?.type)

  // 测试立即创建 blob URL
  if (file.raw) {
    const testUrl = URL.createObjectURL(file.raw)
    console.log('Test blob URL created:', testUrl)
    // 不要忘记清理测试 URL
    setTimeout(() => URL.revokeObjectURL(testUrl), 1000)
  }
}

  const handleVideoError = (type) => {
    if (type === 'enhanced') {
      enhancedVideoError.value = '视频加载失败，请检查网络或视频格式'
    }
  }

const handleUploadError = (error) => {
  console.error('Upload error:', error)
  ElMessage.error('上传失败: ' + (error.message || '网络错误'))
  processing.value = false
  stopStatusPolling()
}

const stopStatusPolling = () => {
  if (statusTimer !== null) {
    clearInterval(statusTimer)
    statusTimer = null
  }
}

const startStatusPolling = (vid) => {
  stopStatusPolling()
  statusTimer = window.setInterval(() => {
    fetchStatus(vid)
  }, 1200)
  fetchStatus(vid)
}

const fetchStatus = async (vid) => {
  if (!vid) return
  try {
    const response = await fetch(`${apiBase}/video/status/${vid}`)
    if (!response.ok) {
      return
    }
    const data = await response.json()
    const current = data.frames_processed || 0
    const total = data.total_frames || 0
    if (total > 0) {
      const percent = Math.min(90, Math.floor((current / total) * 80 + 20))
      progress.value = Math.max(progress.value, percent)
      progressText.value = `处理中：${current}/${total} 帧`
    } else {
      progressText.value = data.status === 'processing' ? '正在增强处理中...' : progressText.value
    }
    if (data.status === 'completed') {
      progress.value = 95
      progressText.value = '增强处理完成，等待最终结果...'
      stopStatusPolling()
    }
    if (data.status === 'failed') {
      progressText.value = '增强处理失败，请重试'
      stopStatusPolling()
    }
  } catch (error) {
    // ignore polling failures silently
  }
}

const handleProcess = async () => {
  await refreshModels()
  if (!modelAvailable(selectedModel.value)) {
    ElMessage.error(unavailableReason(selectedModel.value))
    return
  }
  enhancedVideoUrl.value = ''
  modelChain.value = ''

  if (!videoFile.value) {
    ElMessage.warning('请先选择视频文件')
    return
  }

  processing.value = true
  progress.value = 10
  progressText.value = '正在上传视频...'

  const formData = new FormData()
  formData.append('file', videoFile.value)

  try {
    // Upload video
    const uploadResponse = await fetch(
      `${apiBase}/video/upload_video`,
      {
        method: 'POST',
        body: formData
      }
    )

    if (!uploadResponse.ok) {
      throw new Error((await uploadResponse.json()).detail || '上传失败')
    }

    const uploadData = await uploadResponse.json()
    const vid = uploadData.video_id

    // 设置视频 ID
    videoId.value = vid

    // 使用后端路由获取原始视频（后端会确保编码兼容）
    originalVideoUrl.value = `/video/original/${vid}`
    console.log('Original video URL set to:', originalVideoUrl.value)

    // 清除之前的错误信息
    enhancedVideoError.value = ''

    progress.value = 30
    progressText.value = '视频上传完成，正在增强处理...'

    startStatusPolling(vid)
    await processVideo(vid)

  } catch (error) {
    ElMessage.error('处理失败: ' + error.message)
    processing.value = false
    progress.value = 0
    progressText.value = '处理失败'
    stopStatusPolling()
  }
}

const processVideo = async (vid) => {
    try {
      // 构建查询参数
      const params = new URLSearchParams()
      if (selectedModel.value !== 'auto') {
        params.append('model', selectedModel.value)
      }
      if (postProcessEnabled.value) {
        params.append('post_process', 'true')
      }
      // 始终发送多标签识别参数，让后端知道用户的选择
      params.append('enable_multi_label', multiLabelEnabled.value ? 'true' : 'false')
      params.append('multi_label_threshold', multiLabelThreshold.value.toString())

      const processUrl = `${apiBase}/video/process/${vid}${params.toString() ? `?${params.toString()}` : ''}`

    const processResponse = await fetch(processUrl, { method: 'POST' })

    if (!processResponse.ok) {
      throw new Error((await processResponse.json()).detail || '处理失败')
    }

    const processData = await processResponse.json()

    progress.value = 90
    progressText.value = '处理完成，准备展示...'

      videoId.value = vid
      modelChain.value = processData.model_chain || selectedModel.value
      // 使用相对路径
      enhancedVideoUrl.value = processData.download_url || `/video/download/${vid}`
      if (processData.post_process) {
        progressText.value = '已启用增强后处理，结果已生成。'
      }

    progress.value = 100
    progressText.value = '处理完成!'
    stopStatusPolling()

    // Fetch logs
    await fetchLogs()

    // Emit event
    emit('video-processed', {
      video_id: vid,
      model: selectedModel.value,
      model_chain: modelChain.value
    })

    ElMessage.success('视频增强完成!')

  } catch (error) {
    ElMessage.error('处理失败: ' + error.message)
  } finally {
    stopStatusPolling()
    processing.value = false
  }
}

const fetchLogs = async () => {
  if (!videoId.value) return

  logsLoading.value = true
  try {
    const response = await fetch(`${apiBase}/video/logs/${videoId.value}`)
    const data = await response.json()
    logs.value = data.logs?.join('\n') || '暂无日志'
  } catch (error) {
    logs.value = '获取日志失败: ' + error.message
  } finally {
    logsLoading.value = false
  }
}

const resetUpload = () => {
  videoFile.value = null
  videoId.value = ''
  originalVideoUrl.value = ''
  enhancedVideoUrl.value = ''
  enhancedVideoError.value = ''
  modelChain.value = ''
  logs.value = ''
  progress.value = 0
  progressText.value = '等待处理...'
  multiLabelEnabled.value = false
  multiLabelThreshold.value = 0.3
  stopStatusPolling()
  if (uploadRef.value) {
    uploadRef.value.clearFiles()
  }
}
</script>

<style scoped>
.video-enhancement {
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

/* Model Selection */
.model-desc-text {
  margin-top: 16px;
  text-align: center;
  font-size: 15px;
}

.model-option-label {
  font-weight: 600;
  margin-right: 10px;
  font-size: 18px;
}

.model-option-desc {
  font-size: 16px;
  color: #909399;
}

/* Upload */
.upload-actions {
  display: flex;
  gap: 16px;
  margin-top: 24px;
  justify-content: center;
}

.upload-icon {
  font-size: 64px;
  color: #409eff;
}

.upload-text {
  font-size: 20px;
  color: #606266;
  font-weight: 500;
}

.upload-text em {
  color: #409eff;
  font-style: normal;
  font-weight: 600;
}

.upload-tip {
  font-size: 17px;
  color: #909399;
  margin-top: 10px;
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

.post-process-toggle {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-top: 20px;
  flex-wrap: wrap;
}

.post-process-note {
  color: #606266;
  font-size: 17px;
}

.multi-label-section {
  margin-top: 28px;
  padding: 20px;
  background: #f5f7fa;
  border-radius: 10px;
}

.multi-label-title {
  margin-left: 10px;
  font-size: 20px;
  font-weight: 700;
  color: #303133;
}

.param-desc {
  margin-left: 12px;
  color: #909399;
  font-size: 17px;
}

/* Video Comparison */
.video-comparison {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.video-panel {
  display: flex;
  flex-direction: column;
  gap: 14px;
}

.video-panel h3 {
  font-size: 22px;
  font-weight: 600;
  color: #303133;
  margin: 0;
}

.video-player {
  width: 100%;
  max-height: 450px;
  border-radius: 10px;
  background: #000;
}

.video-placeholder {
  width: 100%;
  height: 350px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  background: #f5f7fa;
  border-radius: 10px;
  color: #909399;
  gap: 16px;
  font-size: 18px;
}

/* Model Info */
.model-info {
  margin-top: 20px;
}

/* Logs */
.logs-container {
  background: #1e1e1e;
  border-radius: 10px;
  padding: 16px;
}

.logs-content {
  font-family: 'Consolas', 'Monaco', monospace;
  font-size: 16px;
  color: #d4d4d4;
  white-space: pre-wrap;
  word-break: break-all;
  margin: 0;
  line-height: 1.6;
}

@media (max-width: 768px) {
  .video-comparison {
    grid-template-columns: 1fr;
  }
}
</style>
