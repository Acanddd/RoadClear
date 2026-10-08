<template>
  <div class="intro-container">
    <div class="intro-content">
      <h1 class="main-title">RoadClear 道路监控视频增强系统</h1>
      <p class="description">
        RoadClear 是一个面向恶劣天气的视频增强系统，可智能调度多种基于深度学习的天气退化网络，
        有效改善雾、雨、雪条件下的道路监控视频画质，增强下游目标检测和车牌识别任务的准确性。
      </p>

      <div class="carousel-section">
        <el-carousel
          ref="carousel"
          :interval="3000"
          type="card"
          height="500px"
          arrow="always"
          indicator-position="outside"
          @mouseenter="handleMouseEnter"
          @mouseleave="handleMouseLeave"
        >
          <el-carousel-item v-for="(img, index) in demoImages" :key="index">
            <div class="image-wrapper">
              <el-image
                :src="img.url"
                fit="contain"
                class="carousel-image"
              >
                <template #placeholder>
                  <div class="image-slot">加载中<span class="dot">...</span></div>
                </template>
              </el-image>
            </div>
          </el-carousel-item>
        </el-carousel>
      </div>

      <div class="action-section">
        <el-button type="primary" size="large" class="start-btn" @click="$emit('start')">
          启动系统
          <el-icon class="el-icon--right"><ArrowRight /></el-icon>
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted } from 'vue'
import { ArrowRight } from '@element-plus/icons-vue'

const emit = defineEmits(['start'])
const carousel = ref(null)
const mouseTimer = ref(null)
const isUserActive = ref(false)

const demoImages = [
  { url: '/demo_images/fog1_aodnet_compare.jpg', label: '雾天增强对比 (AOD-Net)' },
  { url: '/demo_images/fog2_aodnet_compare.jpg', label: '雾天增强对比 (AOD-Net)' },
  { url: '/demo_images/rain1_prenet_compare.jpg', label: '雨天增强对比 (PReNet)' },
  { url: '/demo_images/rain2_prenet_compare.jpg', label: '雨天增强对比 (PReNet)' },
  { url: '/demo_images/snow1_hdcwnet_compare.jpg', label: '雪天增强对比 (TransWeather)' },
  { url: '/demo_images/snow2_hdcwnet_compare.jpg', label: '雪天增强对比 (TransWeather)' }
]

const resetMouseTimer = () => {
  isUserActive.value = true
  if (mouseTimer.value) clearTimeout(mouseTimer.value)

  mouseTimer.value = setTimeout(() => {
    isUserActive.value = false
    // 3秒无动作后，如果轮播已停止则尝试恢复（虽然 el-carousel 默认自动滚动，但我们可以手动确保）
  }, 3000)
}

const handleMouseEnter = () => {
  // 鼠标进入图片区域，el-carousel 默认会暂停 interval
  resetMouseTimer()
}

const handleMouseLeave = () => {
  // 鼠标离开图片区域
  resetMouseTimer()
}

onMounted(() => {
  window.addEventListener('mousemove', resetMouseTimer)
})

onUnmounted(() => {
  window.removeEventListener('mousemove', resetMouseTimer)
  if (mouseTimer.value) clearTimeout(mouseTimer.value)
})
</script>

<style scoped>
.intro-container {
  min-height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 40px 20px;
  background: transparent;
}

.intro-content {
  max-width: 1200px;
  width: 100%;
  text-align: center;
  background: rgba(255, 255, 255, 0.95);
  padding: 40px;
  border-radius: 16px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
  backdrop-filter: blur(10px);
}

.main-title {
  font-size: 36px;
  font-weight: 700;
  color: #303133;
  margin-bottom: 24px;
}

.description {
  font-size: 18px;
  line-height: 1.6;
  color: #606266;
  margin-bottom: 40px;
  text-align: center;
  max-width: 800px;
  margin-left: auto;
  margin-right: auto;
}

.carousel-section {
  margin-bottom: 40px;
}

.image-wrapper {
  position: relative;
  height: 100%;
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: center;
}

.carousel-image {
  width: 100%;
  height: 100%;
  border-radius: 8px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.action-section {
  display: flex;
  justify-content: center;
}

.start-btn {
  font-size: 18px;
  padding: 15px 40px;
  height: auto;
  border-radius: 30px;
  background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
  border: none;
  transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.start-btn:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.image-slot {
  display: flex;
  justify-content: center;
  align-items: center;
  width: 100%;
  height: 100%;
  background: #f5f7fa;
  color: #909399;
}

:deep(.el-carousel__mask) {
  background-color: transparent;
}

:deep(.el-carousel__item--card.is-active) {
  z-index: 2;
}
</style>
