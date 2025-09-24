#!/bin/bash

# TorchServe 自动部署系统安装脚本

set -e

echo "=========================================="
echo "TorchServe 自动部署系统安装"
echo "=========================================="

# 检查Python版本
echo "检查Python版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✓ Python版本符合要求: $python_version"
else
    echo "❌ Python版本过低: $python_version，需要3.8或更高版本"
    exit 1
fi

# 检查pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 未安装"
    exit 1
fi

# 更新pip
echo "更新pip..."
pip3 install --upgrade pip

# 检查是否有CUDA
echo "检查CUDA支持..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ 检测到NVIDIA GPU，将安装CUDA版本的PyTorch"
    CUDA_AVAILABLE=true
else
    echo "⚠️  未检测到NVIDIA GPU，将安装CPU版本的PyTorch"
    CUDA_AVAILABLE=false
fi

# 安装依赖
echo "安装Python依赖包..."

if [ "$CUDA_AVAILABLE" = true ]; then
    # 安装CUDA版本的PyTorch
    echo "安装CUDA版本的PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    # 安装CPU版本的PyTorch
    echo "安装CPU版本的PyTorch..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 安装其他依赖
echo "安装其他依赖包..."
pip3 install -r requirements.txt

# 检查Java（TorchServe需要）
echo "检查Java环境..."
if command -v java &> /dev/null; then
    java_version=$(java -version 2>&1 | awk -F '"' '/version/ {print $2}' | cut -d. -f1)
    if [ "$java_version" -ge 11 ]; then
        echo "✓ Java版本符合要求"
    else
        echo "⚠️  Java版本过低，建议升级到Java 11或更高版本"
    fi
else
    echo "❌ Java未安装，正在安装..."
    # 尝试安装Java（Ubuntu/Debian）
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y openjdk-11-jdk
    elif command -v yum &> /dev/null; then
        sudo yum install -y java-11-openjdk-devel
    else
        echo "请手动安装Java 11或更高版本"
        exit 1
    fi
fi

# 创建工作目录
echo "创建工作目录..."
mkdir -p torchserve_workspace/model_store
mkdir -p torchserve_workspace/temp

# 设置权限
chmod +x torchserve_auto_deploy.py
chmod +x test_deployed_models.py

# 验证安装
echo "验证安装..."
echo "检查TorchServe..."
if python3 -c "import torchserve" 2>/dev/null; then
    echo "✓ TorchServe安装成功"
else
    echo "❌ TorchServe安装失败"
    exit 1
fi

echo "检查torch-model-archiver..."
if command -v torch-model-archiver &> /dev/null; then
    echo "✓ torch-model-archiver安装成功"
else
    echo "❌ torch-model-archiver安装失败"
    exit 1
fi

echo "检查PyTorch..."
if python3 -c "import torch; print(f'PyTorch版本: {torch.__version__}')" 2>/dev/null; then
    echo "✓ PyTorch安装成功"
else
    echo "❌ PyTorch安装失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "🎉 安装完成！"
echo "=========================================="
echo ""
echo "使用方法："
echo "1. 部署模型:"
echo "   python3 torchserve_auto_deploy.py"
echo ""
echo "2. 测试模型:"
echo "   python3 test_deployed_models.py"
echo ""
echo "3. 停止服务:"
echo "   python3 torchserve_auto_deploy.py --stop"
echo ""
echo "4. 清理文件:"
echo "   python3 torchserve_auto_deploy.py --clean"
echo ""
echo "配置文件: deploy_config.yaml"
echo "日志文件: torchserve_deploy.log"
echo ""