# TorchServe 自动部署系统

一个强大的自动化工具，用于部署网上公开的深度学习模型到TorchServe，支持多种模型源和自动化测试。

## 🌟 特性

- **多模型源支持**: HuggingFace、TorchVision、PyTorch Hub、本地模型
- **自动化部署**: 一键下载、转换、部署模型
- **智能测试**: 自动化模型功能和性能测试
- **灵活配置**: YAML配置文件，易于定制
- **完整日志**: 详细的部署和测试日志
- **快速启动**: 简化脚本，5分钟内完成部署

## 📋 系统要求

- Python 3.8+
- Java 11+ (TorchServe依赖)
- 8GB+ RAM (建议)
- CUDA (可选，GPU加速)

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆或下载脚本
chmod +x install.sh
./install.sh
```

### 2. 快速部署单个模型

```bash
# 使用快速启动脚本部署ResNet18
python3 quick_start.py
```

### 3. 部署多个模型

```bash
# 使用完整部署脚本
python3 torchserve_auto_deploy.py
```

### 4. 测试部署的模型

```bash
# 测试所有模型
python3 test_deployed_models.py

# 测试特定模型
python3 test_deployed_models.py --model resnet18

# 性能测试
python3 test_deployed_models.py --performance
```

## 📁 项目结构

```
torchserve-auto-deploy/
├── torchserve_auto_deploy.py    # 主要部署脚本
├── test_deployed_models.py      # 测试脚本
├── quick_start.py               # 快速启动脚本
├── deploy_config.yaml           # 部署配置文件
├── requirements.txt             # Python依赖
├── install.sh                   # 安装脚本
├── README.md                    # 文档
└── torchserve_workspace/        # 工作目录
    ├── model_store/             # 模型存储
    ├── temp/                    # 临时文件
    └── config.properties        # TorchServe配置
```

## ⚙️ 配置文件说明

`deploy_config.yaml` 文件包含所有部署配置：

```yaml
# 工作目录配置
work_dir: "./torchserve_workspace"

# TorchServe服务配置
torchserve:
  inference_address: "http://127.0.0.1:8080"
  management_address: "http://127.0.0.1:8081"
  initial_workers: 1
  max_workers: 4

# 模型配置
models:
  - name: "resnet18"
    source: "torchvision"
    model_name: "resnet18"
    pretrained: true
    handler: "image_classifier"
    version: "1.0"
```

### 支持的模型源

#### 1. TorchVision模型

```yaml
- name: "resnet50"
  source: "torchvision"
  model_name: "resnet50"
  pretrained: true
  handler: "image_classifier"
```

#### 2. HuggingFace模型

```yaml
- name: "bert_base"
  source: "huggingface"
  hf_model_name: "bert-base-uncased"
  handler: "custom"
```

#### 3. PyTorch Hub模型

```yaml
- name: "yolov5s"
  source: "pytorch_hub"
  repo: "ultralytics/yolov5"
  model_name: "yolov5s"
  handler: "object_detector"
```

#### 4. 本地模型

```yaml
- name: "custom_model"
  source: "local"
  model_path: "/path/to/model.pt"
  config_path: "/path/to/config.json"
  handler: "custom"
```

## 🔧 使用方法

### 基本命令

```bash
# 完整部署流程
python3 torchserve_auto_deploy.py

# 使用自定义配置
python3 torchserve_auto_deploy.py --config my_config.yaml

# 停止TorchServe服务
python3 torchserve_auto_deploy.py --stop

# 清理工作目录
python3 torchserve_auto_deploy.py --clean
```

### 测试命令

```bash
# 测试所有模型
python3 test_deployed_models.py

# 测试特定模型
python3 test_deployed_models.py --model resnet18

# 性能测试
python3 test_deployed_models.py --performance

# 列出已注册模型
python3 test_deployed_models.py --list
```

### 快速启动命令

```bash
# 快速部署ResNet18
python3 quick_start.py

# 仅测试模型
python3 quick_start.py --test-only

# 停止服务
python3 quick_start.py --stop

# 清理文件
python3 quick_start.py --clean
```

## 🧪 测试功能

### 自动化测试

脚本会根据模型类型自动选择合适的测试方法：

- **图像分类模型**: 使用随机生成的图像进行分类测试
- **文本模型**: 使用预定义的文本进行处理测试
- **目标检测模型**: 使用图像进行物体检测测试
- **通用模型**: 使用数值数据进行推理测试

### 性能测试

```bash
# 性能测试包括：
# - 多次请求测试
# - 响应时间统计
# - 成功率计算
# - 并发测试
python3 test_deployed_models.py --performance
```

### 测试结果

测试脚本会输出详细的测试报告：

```
========================================
测试模型: resnet18
========================================
✓ resnet18 预测成功
  预测结果: {'class_1': 0.23, 'class_2': 0.45, ...}

性能测试结果:
  成功率: 100.0% (10/10)
  平均响应时间: 0.15s
  总时间: 1.52s
```

## 🔌 API使用

部署完成后，可以通过REST API使用模型：

### 推理API

```bash
# 图像分类
curl -X POST http://localhost:8080/predictions/resnet18 \
     -F "data=@image.jpg"

# 文本处理
curl -X POST http://localhost:8080/predictions/bert_base \
     -H "Content-Type: application/json" \
     -d '{"text": "Hello world"}'
```

### 管理API

```bash
# 查看所有模型
curl http://localhost:8081/models

# 查看模型详情
curl http://localhost:8081/models/resnet18

# 扩缩容
curl -X PUT http://localhost:8081/models/resnet18 \
     -d "min_worker=2&max_worker=4"
```

## 🐛 故障排除

### 常见问题

1. **Java版本过低**
   ```bash
   # 安装Java 11+
   sudo apt install openjdk-11-jdk
   ```

2. **端口被占用**
   ```bash
   # 检查端口占用
   lsof -i :8080
   lsof -i :8081
   
   # 修改配置文件中的端口
   ```

3. **内存不足**
   ```bash
   # 减少worker数量
   # 在配置文件中设置：
   initial_workers: 1
   max_workers: 2
   ```

4. **CUDA相关错误**
   ```bash
   # 安装CPU版本的PyTorch
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

### 日志查看

```bash
# 查看部署日志
tail -f torchserve_deploy.log

# 查看TorchServe日志
tail -f torchserve_workspace/logs/ts_log.log
```

## 🤝 贡献

欢迎提交Issues和Pull Requests！

1. Fork 项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开Pull Request

## 📄 许可证

本项目使用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [TorchServe](https://pytorch.org/serve/) - PyTorch模型服务框架
- [HuggingFace](https://huggingface.co/) - 预训练模型库
- [PyTorch](https://pytorch.org/) - 深度学习框架

---

如果这个项目对您有帮助，请给个⭐️！