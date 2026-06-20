@echo off
chcp 65001 >nul
echo ========================================
echo RoadClear 系统安装脚本
echo ========================================
echo.

:: 检查Python是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.8-3.11
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/5] 检测到Python版本:
python --version
echo.

:: 创建虚拟环境
echo [2/5] 创建Python虚拟环境...
if exist venv (
    echo 虚拟环境已存在，跳过创建
) else (
    python -m venv venv
    if errorlevel 1 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
    echo 虚拟环境创建成功
)
echo.

:: 激活虚拟环境
echo [3/5] 激活虚拟环境...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [错误] 激活虚拟环境失败
    pause
    exit /b 1
)
echo.

:: 升级pip
echo [4/5] 升级pip到最新版本...
python -m pip install --upgrade pip
echo.

:: 安装依赖
echo [5/5] 安装项目依赖（这可能需要几分钟）...
echo.
echo 提示: 如果您有NVIDIA GPU，可以手动安装GPU版本的PyTorch:
echo   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
echo.
pip install -r requirements.txt
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，请检查错误信息
    echo 您可以尝试手动安装: pip install -r requirements.txt
) else (
    echo.
    echo ========================================
    echo 安装完成！
    echo ========================================
    echo.
    echo 下一步:
    echo 1. 运行 start_backend.bat 启动后端服务
    echo 2. 打开浏览器访问 http://localhost:8000
    echo.
)

pause
