@echo off
chcp 65001 >nul
echo ========================================
echo RoadClear 前端开发服务器启动脚本
echo ========================================
echo.
echo 注意: 生产环境不需要运行此脚本
echo 前端已构建为静态文件，通过后端服务提供
echo.
echo 此脚本仅用于前端开发和调试
echo ========================================
echo.

:: 检查Node.js
node --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Node.js
    echo 前端已构建，无需Node.js即可运行
    echo 请直接使用 start_backend.bat 启动系统
    pause
    exit /b 1
)

echo Node.js版本:
node --version
echo.

:: 检查前端源码
if not exist frontend\src (
    echo [错误] 前端源码不存在
    echo 此部署包仅包含构建后的前端文件
    echo 如需开发，请使用完整的开发环境
    pause
    exit /b 1
)

:: 安装依赖
if not exist frontend\node_modules (
    echo 安装前端依赖...
    cd frontend
    npm install
    cd ..
)

:: 启动开发服务器
echo 启动前端开发服务器...
cd frontend
npm run dev

pause
