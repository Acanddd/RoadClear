<template>
  <div class="parameter-console">
    <el-card class="console-card" shadow="hover">
      <template #header>
        <div class="card-header">
          <span><el-icon><Setting /></el-icon> 参数控制台</span>
          <div class="header-actions">
            <el-button size="small" @click="loadConfig" :loading="loading">
              <el-icon><Refresh /></el-icon> 刷新
            </el-button>
            <el-button size="small" type="warning" @click="resetToDefaults" :loading="loading">
              <el-icon><RefreshLeft /></el-icon> 重置默认
            </el-button>
            <el-button size="small" type="primary" @click="saveConfig" :loading="saving">
              <el-icon><Check /></el-icon> 保存配置
            </el-button>
          </div>
        </div>
      </template>

      <el-tabs v-model="activeTab" @tab-click="handleTabClick">
        <!-- 模型参数 -->
        <el-tab-pane label="模型参数" name="models">
          <div class="param-section">
            <h3>AOD-Net (去雾模型)</h3>
            <el-form :model="config.models.aodnet" label-width="120px">
              <el-form-item label="使用 ONNX">
                <el-switch disabled v-model="config.models.aodnet.use_onnx" />
              </el-form-item>
            </el-form>
          </div>

          <div class="param-section">
            <h3>PreNet (去雨模型)</h3>
            <el-form :model="config.models.prenet" label-width="120px">
              <el-form-item label="递归迭代次数">
                <el-slider
                  disabled v-model="config.models.prenet.recurrent_iter"
                  :min="1"
                  :max="20"
                  :step="1"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
              <el-form-item label="使用 ONNX">
                <el-switch disabled v-model="config.models.prenet.use_onnx" />
              </el-form-item>
            </el-form>
          </div>

          <div class="param-section">
            <h3>TransWeather (去雪模型)</h3>
            <el-form :model="config.models.hdcwnet" label-width="120px">
              <el-form-item label="使用 ONNX">
                <el-switch disabled v-model="config.models.hdcwnet.use_gpu" />
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <!-- 后处理参数 -->
        <el-tab-pane label="后处理参数" name="postprocessing">
          <div class="param-section">
            <h3>CLAHE (对比度增强)</h3>
            <el-form :model="config.postprocessing.clahe" label-width="120px">
              <el-form-item label="启用">
                <el-switch v-model="config.postprocessing.clahe.enabled" />
              </el-form-item>
              <el-form-item label="对比度限制">
                <el-slider
                  v-model="config.postprocessing.clahe.clip_limit"
                  :min="0.1"
                  :max="10.0"
                  :step="0.1"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
              <el-form-item label="网格大小">
                <el-input-number
                  v-model="config.postprocessing.clahe.tile_grid_size[0]"
                  :min="1"
                  :max="32"
                  :step="1"
                  style="width: 100px"
                />
                <span style="margin: 0 10px">x</span>
                <el-input-number
                  v-model="config.postprocessing.clahe.tile_grid_size[1]"
                  :min="1"
                  :max="32"
                  :step="1"
                  style="width: 100px"
                />
              </el-form-item>
            </el-form>
          </div>

          <div class="param-section">
            <h3>非锐化掩膜 (边缘增强)</h3>
            <el-form :model="config.postprocessing.unsharp_mask" label-width="120px">
              <el-form-item label="启用">
                <el-switch v-model="config.postprocessing.unsharp_mask.enabled" />
              </el-form-item>
              <el-form-item label="核大小">
                <el-input-number
                  v-model="config.postprocessing.unsharp_mask.kernel_size[0]"
                  :min="1"
                  :max="15"
                  :step="2"
                  style="width: 100px"
                />
                <span style="margin: 0 10px">x</span>
                <el-input-number
                  v-model="config.postprocessing.unsharp_mask.kernel_size[1]"
                  :min="1"
                  :max="15"
                  :step="2"
                  style="width: 100px"
                />
              </el-form-item>
              <el-form-item label="高斯标准差">
                <el-slider
                  v-model="config.postprocessing.unsharp_mask.sigma"
                  :min="0.1"
                  :max="10.0"
                  :step="0.1"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
              <el-form-item label="锐化强度">
                <el-slider
                  v-model="config.postprocessing.unsharp_mask.amount"
                  :min="0.1"
                  :max="5.0"
                  :step="0.1"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
              <el-form-item label="阈值">
                <el-slider
                  v-model="config.postprocessing.unsharp_mask.threshold"
                  :min="0"
                  :max="255"
                  :step="1"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
            </el-form>
          </div>

          <div class="param-section">
            <h3>降噪 (双边滤波)</h3>
            <el-form :model="config.postprocessing.denoise" label-width="120px">
              <el-form-item label="启用">
                <el-switch v-model="config.postprocessing.denoise.enabled" />
              </el-form-item>
              <el-form-item label="滤波直径">
                <el-slider
                  v-model="config.postprocessing.denoise.diameter"
                  :min="1"
                  :max="15"
                  :step="1"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
              <el-form-item label="颜色空间标准差">
                <el-slider
                  v-model="config.postprocessing.denoise.sigma_color"
                  :min="1.0"
                  :max="200.0"
                  :step="1.0"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
              <el-form-item label="坐标空间标准差">
                <el-slider
                  v-model="config.postprocessing.denoise.sigma_space"
                  :min="1.0"
                  :max="200.0"
                  :step="1.0"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
            </el-form>
          </div>

          <div class="param-section">
            <h3>动态范围压缩</h3>
            <el-form :model="config.postprocessing.dynamic_range" label-width="120px">
              <el-form-item label="启用">
                <el-switch v-model="config.postprocessing.dynamic_range.enabled" />
              </el-form-item>
              <el-form-item label="亮度阈值">
                <el-slider
                  v-model="config.postprocessing.dynamic_range.bright_threshold"
                  :min="0"
                  :max="255"
                  :step="1"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
              <el-form-item label="压缩斜率">
                <el-slider
                  v-model="config.postprocessing.dynamic_range.slope"
                  :min="0.1"
                  :max="1.0"
                  :step="0.01"
                  show-input
                  style="width: 300px"
                />
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>

        <!-- 调度器参数 -->
        <el-tab-pane label="调度器参数" name="dispatcher">
          <div class="param-section">
            <el-form :model="config.dispatcher" label-width="120px">
              <el-form-item label="FPS 阈值">
                <el-slider
                  v-model="config.dispatcher.fps_threshold"
                  :min="0.1"
                  :max="60.0"
                  :step="0.1"
                  show-input
                  style="width: 300px"
                />
                <span class="param-desc">低于此 FPS 时自动降级为简单增强</span>
              </el-form-item>
              <el-form-item label="天气分类跳帧间隔">
                <el-slider
                  v-model="config.dispatcher.frame_skip_interval"
                  :min="1"
                  :max="300"
                  :step="1"
                  show-input
                  style="width: 300px"
                />
                <span class="param-desc">每 N 帧进行一次天气分类以提高性能</span>
              </el-form-item>
              <el-form-item label="EMA 平滑系数">
                <el-slider
                  v-model="config.dispatcher.ema_alpha"
                  :min="0.01"
                  :max="1.0"
                  :step="0.01"
                  show-input
                  style="width: 300px"
                />
                <span class="param-desc">FPS 指数移动平均的平滑系数，越小越平滑</span>
              </el-form-item>
            </el-form>
          </div>
        </el-tab-pane>
      </el-tabs>

      <div class="console-footer">
        <el-alert
          title="参数修改说明"
          description="修改参数后需要点击保存配置按钮，参数将在下次视频处理时生效。某些参数修改可能需要重启后端服务。"
          type="info"
          :closable="false"
          show-icon
        />
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import {
  Setting,
  Refresh,
  RefreshLeft,
  Check
} from '@element-plus/icons-vue'

const props = defineProps({
  backendUrl: {
    type: String,
    default: ''  // 使用相对路径，通过 Vite 代理
  }
})

const emit = defineEmits(['config-updated'])

const activeTab = ref('models')
const loading = ref(false)
const saving = ref(false)

// 默认配置结构
const defaultConfig = {
  models: {
    aodnet: {
      use_onnx: true
    },
    prenet: {
      recurrent_iter: 6,
      use_onnx: true
    },
    hdcwnet: {
      use_gpu: false
    }
  },
  postprocessing: {
    clahe: {
      enabled: true,
      clip_limit: 1.5,
      tile_grid_size: [8, 8]
    },
    unsharp_mask: {
      enabled: true,
      kernel_size: [5, 5],
      sigma: 1.0,
      amount: 0.8,
      threshold: 5
    },
    denoise: {
      enabled: true,
      diameter: 5,
      sigma_color: 50.0,
      sigma_space: 50.0
    },
    dynamic_range: {
      enabled: true,
      bright_threshold: 230,
      slope: 0.6
    }
  },
  dispatcher: {
    fps_threshold: 5.0,
    frame_skip_interval: 30,
    ema_alpha: 0.1,
    enable_multi_label: false,
    multi_label_threshold: 0.3
  }
}

const config = reactive(JSON.parse(JSON.stringify(defaultConfig)))

const apiBase = ''

const formatThresholdTooltip = (value) => {
  return `${(value * 100).toFixed(0)}%`
}

const saveConfig = async () => {
  saving.value = true
  try {
    const response = await fetch(`${apiBase}/config/config`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(config)
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const result = await response.json()
    ElMessage.success('配置保存成功')
    emit('config-updated', result)
  } catch (error) {
    console.error('Save config error:', error)
    ElMessage.error('保存配置失败: ' + error.message)
  } finally {
    saving.value = false
  }
}

const resetToDefaults = async () => {
  try {
    await ElMessageBox.confirm(
      '确定要重置所有参数为默认值吗？此操作不可撤销。',
      '确认重置',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning'
      }
    )

    loading.value = true
    const response = await fetch(`${apiBase}/config/config/reset`, {
      method: 'POST'
    })

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }

    const result = await response.json()
    Object.assign(config, result)
    ElMessage.success('已重置为默认配置')
    emit('config-updated', result)
  } catch (error) {
    if (error !== 'cancel') {
      console.error('Reset config error:', error)
      ElMessage.error('重置配置失败: ' + error.message)
    }
  } finally {
    loading.value = false
  }
}

const loadConfig = async () => {
  loading.value = true
  try {
    const response = await fetch(`${apiBase}/config/config`)
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    const data = await response.json()

    // 深度合并配置
    Object.assign(config, data)
    ElMessage.success('配置加载成功')
  } catch (error) {
    console.error('Load config error:', error)
    ElMessage.error('加载配置失败: ' + error.message)
  } finally {
    loading.value = false
  }
}

const handleTabClick = (tab) => {
  // 可以在这里添加标签切换时的逻辑
}

onMounted(() => {
  loadConfig()
})
</script>

<style scoped>
.parameter-console {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: 600;
}

.header-actions {
  display: flex;
  gap: 8px;
}

.param-section {
  margin-bottom: 30px;
  padding: 20px;
  border: 1px solid #e4e7ed;
  border-radius: 8px;
  background: #fafafa;
}

.param-section h3 {
  margin: 0 0 20px 0;
  color: #303133;
  font-size: 16px;
  font-weight: 600;
  display: flex;
  align-items: center;
  gap: 8px;
}

.param-desc {
  margin-left: 10px;
  color: #909399;
  font-size: 12px;
}

.threshold-explanation {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  font-size: 13px;
  color: #606266;
}

.console-footer {
  margin-top: 20px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .header-actions {
    flex-direction: column;
    align-items: stretch;
  }

  .param-section {
    padding: 15px;
  }
}
</style>
