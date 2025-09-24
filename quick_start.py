#!/usr/bin/env python3
"""
TorchServe 快速启动脚本
简化版本，快速部署和测试常用模型
"""

import os
import sys
import subprocess
import time
import requests
import argparse
from pathlib import Path

def run_command(cmd, check=True):
    """运行命令"""
    print(f"执行: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"命令执行失败: {result.stderr}")
        sys.exit(1)
    return result

def check_dependencies():
    """检查依赖"""
    print("检查依赖...")
    
    try:
        import torch
        import torchvision
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError:
        print("❌ PyTorch未安装")
        return False
    
    # 检查torch-model-archiver
    result = run_command(['which', 'torch-model-archiver'], check=False)
    if result.returncode == 0:
        print("✓ torch-model-archiver")
    else:
        print("❌ torch-model-archiver未安装")
        return False
    
    # 检查torchserve
    result = run_command(['which', 'torchserve'], check=False)
    if result.returncode == 0:
        print("✓ torchserve")
    else:
        print("❌ torchserve未安装")
        return False
    
    return True

def create_simple_handler():
    """创建简单的图像分类处理器"""
    handler_content = '''
import torch
import torchvision.transforms as transforms
from torchvision import models
from ts.torch_handler.image_classifier import ImageClassifier

class SimpleImageClassifier(ImageClassifier):
    def __init__(self):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
'''
    
    handler_path = Path("simple_handler.py")
    with open(handler_path, 'w') as f:
        f.write(handler_content)
    
    return str(handler_path)

def create_model():
    """创建并保存ResNet18模型"""
    print("创建ResNet18模型...")
    
    import torch
    import torchvision.models as models
    
    # 创建模型
    model = models.resnet18(pretrained=True)
    model.eval()
    
    # 保存模型
    model_path = Path("resnet18.pt")
    torch.jit.script(model).save(model_path)
    
    print(f"✓ 模型已保存: {model_path}")
    return str(model_path)

def create_mar_file(model_path, handler_path):
    """创建MAR文件"""
    print("创建MAR文件...")
    
    # 创建model_store目录
    model_store = Path("model_store")
    model_store.mkdir(exist_ok=True)
    
    cmd = [
        'torch-model-archiver',
        '--model-name', 'resnet18',
        '--version', '1.0',
        '--serialized-file', model_path,
        '--handler', handler_path,
        '--export-path', str(model_store)
    ]
    
    run_command(cmd)
    
    mar_file = model_store / "resnet18.mar"
    print(f"✓ MAR文件已创建: {mar_file}")
    return str(mar_file)

def start_torchserve():
    """启动TorchServe"""
    print("启动TorchServe...")
    
    # 创建配置文件
    config_content = '''
inference_address=http://127.0.0.1:8080
management_address=http://127.0.0.1:8081
model_store=model_store
initial_workers=1
max_workers=2
'''
    
    config_path = Path("config.properties")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    # 启动服务
    cmd = ['torchserve', '--start', '--ts-config', str(config_path), '--ncs']
    subprocess.Popen(cmd)
    
    # 等待服务启动
    print("等待服务启动...")
    for i in range(30):
        try:
            response = requests.get("http://127.0.0.1:8080/ping", timeout=2)
            if response.status_code == 200:
                print("✓ TorchServe已启动")
                return True
        except:
            pass
        time.sleep(1)
        print(f"等待中... ({i+1}/30)")
    
    print("❌ TorchServe启动失败")
    return False

def register_model(mar_file):
    """注册模型"""
    print("注册模型...")
    
    url = "http://127.0.0.1:8081/models"
    params = {
        'url': mar_file,
        'initial_workers': 1,
        'synchronous': 'true'
    }
    
    try:
        response = requests.post(url, params=params, timeout=30)
        if response.status_code == 200:
            print("✓ 模型注册成功")
            return True
        else:
            print(f"❌ 模型注册失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 模型注册失败: {e}")
        return False

def test_model():
    """测试模型"""
    print("测试模型...")
    
    from PIL import Image
    import numpy as np
    
    # 创建测试图像
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    img_path = Path("test_image.jpg")
    img.save(img_path)
    
    try:
        url = "http://127.0.0.1:8080/predictions/resnet18"
        with open(img_path, 'rb') as f:
            files = {'data': f}
            response = requests.post(url, files=files, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            print("✓ 模型测试成功")
            print(f"预测结果: {result}")
            return True
        else:
            print(f"❌ 模型测试失败: {response.text}")
            return False
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")
        return False
    finally:
        # 清理测试图像
        if img_path.exists():
            img_path.unlink()

def stop_torchserve():
    """停止TorchServe"""
    print("停止TorchServe...")
    try:
        run_command(['torchserve', '--stop'], check=False)
        print("✓ TorchServe已停止")
    except:
        print("⚠️  停止TorchServe时出现错误")

def cleanup():
    """清理文件"""
    print("清理临时文件...")
    files_to_remove = [
        "resnet18.pt",
        "simple_handler.py",
        "test_image.jpg",
        "config.properties"
    ]
    
    for file in files_to_remove:
        path = Path(file)
        if path.exists():
            path.unlink()
            print(f"已删除: {file}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TorchServe快速启动')
    parser.add_argument('--stop', action='store_true', help='停止服务')
    parser.add_argument('--clean', action='store_true', help='清理文件')
    parser.add_argument('--test-only', action='store_true', help='仅测试模型')
    
    args = parser.parse_args()
    
    if args.stop:
        stop_torchserve()
        return
    
    if args.clean:
        cleanup()
        return
    
    if args.test_only:
        if test_model():
            print("\n🎉 测试成功！")
        else:
            print("\n❌ 测试失败！")
        return
    
    print("========================================")
    print("TorchServe 快速启动")
    print("========================================")
    
    # 1. 检查依赖
    if not check_dependencies():
        print("\n请先运行 ./install.sh 安装依赖")
        sys.exit(1)
    
    try:
        # 2. 创建处理器
        handler_path = create_simple_handler()
        
        # 3. 创建模型
        model_path = create_model()
        
        # 4. 创建MAR文件
        mar_file = create_mar_file(model_path, handler_path)
        
        # 5. 启动TorchServe
        if not start_torchserve():
            sys.exit(1)
        
        # 6. 注册模型
        if not register_model(mar_file):
            sys.exit(1)
        
        # 7. 测试模型
        if test_model():
            print("\n🎉 快速启动成功！")
            print("\n使用说明:")
            print("- 推理API: http://127.0.0.1:8080")
            print("- 管理API: http://127.0.0.1:8081")
            print("- 测试命令: python3 quick_start.py --test-only")
            print("- 停止服务: python3 quick_start.py --stop")
            print("- 清理文件: python3 quick_start.py --clean")
        else:
            print("\n❌ 模型测试失败")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n用户中断，正在清理...")
        stop_torchserve()
        cleanup()
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        stop_torchserve()
        cleanup()
        sys.exit(1)

if __name__ == '__main__':
    main()