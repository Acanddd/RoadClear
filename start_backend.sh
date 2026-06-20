#!/bin/bash

echo "========================================"
echo "RoadClear 后端服务启动脚本"
echo "========================================"
echo ""

# 检查虚拟环境
if [ ! -d "venv" ]; then
    echo "[错误] 虚拟环境不存在，请先运行 ./install.sh"
    exit 1
fi

# 激活虚拟环境
echo "[1/3] 激活虚拟环境..."
source venv/bin/activate
echo ""

# 检查模型文件
echo "[2/3] 检查模型文件..."
if [ ! -f "backend/runs/detrac_yolov5s_4class/weights/best.pt" ]; then
    echo "[警告] 车辆检测模型未找到"
fi
if [ ! -f "backend/runs/ccpd_yolov8/weights/best.pt" ]; then
    echo "[警告] 车牌识别模型未找到"
fi
echo ""

# 启动服务
echo "[3/3] 启动后端服务..."
echo ""
echo "服务将运行在: http://localhost:8000"
echo "API文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止服务"
echo "========================================"
echo ""

cd backend
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
