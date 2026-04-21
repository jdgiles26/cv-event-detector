#!/bin/bash
# macOS Compatible CV Event Detector Startup Script
# Runs on Apple Silicon (M1/M2/M3/M4) using Metal GPU

set -e

echo "============================================================"
echo "  CV Event Detector - macOS Version"
echo "============================================================"

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "⚠️  Warning: This script is designed for macOS. Detected: $OSTYPE"
fi

# Check for Apple Silicon
ARCH=$(uname -m)
if [[ "$ARCH" == "arm64" ]]; then
    echo -e "${GREEN}✅ Apple Silicon (ARM64) detected${NC}"
else
    echo -e "${YELLOW}ℹ️  Intel Mac detected${NC}"
fi

# Create necessary directories
echo -e "${BLUE}📁 Creating output directories...${NC}"
mkdir -p /tmp/via/output
mkdir -p /tmp/via-logs
mkdir -p ~/.via/ngc_model_cache/cv_models

# Check Python version
echo -e "${BLUE}🐍 Checking Python...${NC}"
python3 --version

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
echo -e "${BLUE}🚀 Activating virtual environment...${NC}"
source venv/bin/activate

# Install/update requirements
echo -e "${BLUE}📥 Installing dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements_macos.txt

# Check PyTorch MPS availability
echo -e "${BLUE}🔍 Checking PyTorch Metal (MPS) support...${NC}"
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
if torch.backends.mps.is_available():
    print('✅ Metal (MPS) is available - GPU acceleration enabled!')
else:
    print('ℹ️  Metal (MPS) not available - will use CPU')
"

# Download YOLO model if not present
if [ ! -f "$HOME/.via/ngc_model_cache/cv_models/yolov8n.onnx" ]; then
    echo -e "${BLUE}📥 Downloading YOLOv8 model (one-time)...${NC}"
    mkdir -p ~/.via/ngc_model_cache/cv_models
    curl -L -o ~/.via/ngc_model_cache/cv_models/yolov8n.onnx \
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
    echo -e "${GREEN}✅ Model downloaded${NC}"
fi

echo ""
echo "============================================================"
echo -e "${GREEN}🎉 Starting CV Event Detector on macOS${NC}"
echo ""
echo "📱 Web UI:     http://localhost:8000"
echo "📖 API Docs:   http://localhost:8000/docs"
echo "🏥 Health:     http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop"
echo "============================================================"
echo ""

# Run the application
python3 main_macos.py
