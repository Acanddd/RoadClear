<template>
  <div class="app-container">
    <!-- Intro Page -->
    <IntroPage v-if="showIntro" @start="showIntro = false" />

    <el-container v-else>
      <!-- Header -->
      <el-header class="app-header">
        <div class="header-content">
          <div class="logo-section">
            <el-icon class="logo-icon" :size="32"><VideoCameraFilled /></el-icon>
            <h1 class="app-title">RoadClear</h1>
            <span class="app-subtitle">道路监控视频增强系统</span>
          </div>
          <div class="header-actions">
            <el-tag :type="backendConnected ? 'success' : 'danger'" effect="plain">
              <el-icon><Connection /></el-icon>
              后端状态: {{ backendConnected ? '已连接' : '未连接' }}
            </el-tag>
          </div>
        </div>
      </el-header>

      <!-- Main Content -->
      <el-main class="app-main">
        <el-tabs v-model="activeTab" type="border-card" class="main-tabs">
          <!-- Video Enhancement Tab -->
            <el-tab-pane label="视频增强" name="enhance">
              <VideoEnhancement
                :backend-url="''"
                @video-processed="handleVideoProcessed"
              />
            </el-tab-pane>

          <!-- Algorithm Showcase Tab -->
          <el-tab-pane label="增强算法展示" name="showcase">
            <AlgorithmShowcase />
          </el-tab-pane>

          <!-- Parameter Console Tab -->
          <el-tab-pane label="参数控制台" name="console">
            <ParameterConsole
              :backend-url="backendUrl"
              @config-updated="handleConfigUpdated"
            />
          </el-tab-pane>

          <!-- Evaluation Tab -->
          <el-tab-pane label="任务评估" name="evaluation">
            <VideoEvaluation
              :backend-url="backendUrl"
              :current-video-id="currentVideoId"
            />
          </el-tab-pane>
        </el-tabs>
      </el-main>

      <!-- Footer -->
      <el-footer class="app-footer">
        <p>RoadClear v1.0 - 面向恶劣天气的道路监控视频增强系统</p>
      </el-footer>
    </el-container>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { VideoCameraFilled, Connection } from '@element-plus/icons-vue'
import IntroPage from './components/IntroPage.vue'
import VideoEnhancement from './components/VideoEnhancement.vue'
import AlgorithmShowcase from './components/AlgorithmShowcase.vue'
import VideoEvaluation from './components/VideoEvaluation.vue'
import ParameterConsole from './components/ParameterConsole.vue'

const backendUrl = ref('')  // 使用空字符串，通过 Vite 代理访问后端
const activeTab = ref('enhance')
const backendConnected = ref(false)
const currentVideoId = ref('')
const showIntro = ref(true)

const checkBackendConnection = async () => {
  try {
    // 使用 /health 端点检查后端连接
    const response = await fetch('/health')
    if (response.ok) {
      backendConnected.value = true
    }
  } catch (error) {
    backendConnected.value = false
  }
}

const handleVideoProcessed = (data) => {
  currentVideoId.value = data.video_id
  activeTab.value = 'evaluation'
}

const handleConfigUpdated = () => {
  // 参数更新后的处理逻辑，可以在这里添加通知或其他操作
  console.log('Configuration updated')
}

onMounted(() => {
  checkBackendConnection()
  setInterval(checkBackendConnection, 30000)
})
</script>

<style scoped>
.app-container {
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}

.el-container {
  min-height: 100vh;
}

.app-header {
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.1);
  padding: 0 20px;
  display: flex;
  align-items: center;
}

.header-content {
  width: 100%;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.logo-section {
  display: flex;
  align-items: center;
  gap: 12px;
}

.logo-icon {
  color: #667eea;
}

.app-title {
  font-size: 24px;
  font-weight: 700;
  color: #303133;
  margin: 0;
}

.app-subtitle {
  font-size: 24px;
  font-weight: 700;
  color: #303133;
  padding-left: 12px;
  border-left: 2px solid #dcdfe6;
}

.header-actions {
  display: flex;
  align-items: center;
  gap: 12px;
}

.app-main {
  padding: 20px;
}

.main-tabs {
  max-width: 1600px;
  margin: 0 auto;
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
}

.app-footer {
  background: rgba(255, 255, 255, 0.9);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 16px;
}

.app-footer p {
  color: #909399;
  font-size: 14px;
  margin: 0;
}

:deep(.el-tabs__header) {
  background: #f5f7fa;
}

:deep(.el-tabs__item) {
  font-size: 20px;
  font-weight: 700;
  padding: 0 24px;
}

:deep(.el-tabs__item.is-active) {
  color: #667eea;
}

:deep(.el-tabs__active-bar) {
  background-color: #667eea;
}
</style>
