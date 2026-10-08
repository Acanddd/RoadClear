import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const target = env.ROADCLEAR_BACKEND_URL || 'http://127.0.0.1:8000'
  const proxy = Object.fromEntries(['/health','/ready','/video','/videos','/eval','/models','/config','/detection','/processed','/postprocessed','/demo_images'].map(path => [path, { target, changeOrigin: true, ws: true, timeout: 0, proxyTimeout: 0 }]))
  return { plugins: [vue()], server: { host: '127.0.0.1', port: 5173, strictPort: true, proxy },
           preview: { host: '127.0.0.1', port: 5173, strictPort: true, proxy } }
})
