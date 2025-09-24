#!/usr/bin/env python3
"""
TorchServe 自动部署脚本
支持从多个源（HuggingFace, PyTorch Hub, 本地文件）下载模型并自动部署到TorchServe
"""

import os
import sys
import json
import shutil
import subprocess
import tempfile
import time
import requests
import yaml
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('torchserve_deploy.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TorchServeDeployer:
    """TorchServe 自动部署器"""
    
    def __init__(self, config_file: str = "deploy_config.yaml"):
        """
        初始化部署器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
        self.work_dir = Path(self.config.get('work_dir', './torchserve_workspace'))
        self.model_store = self.work_dir / 'model_store'
        self.temp_dir = self.work_dir / 'temp'
        self.torchserve_config = self.work_dir / 'config.properties'
        
        # 创建工作目录
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.model_store.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        # TorchServe进程
        self.torchserve_process = None
        
    def _load_config(self) -> Dict:
        """加载配置文件"""
        if not os.path.exists(self.config_file):
            logger.warning(f"配置文件 {self.config_file} 不存在，使用默认配置")
            return self._get_default_config()
        
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"成功加载配置文件: {self.config_file}")
            return config
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """获取默认配置"""
        return {
            'work_dir': './torchserve_workspace',
            'torchserve': {
                'inference_address': 'http://127.0.0.1:8080',
                'management_address': 'http://127.0.0.1:8081',
                'model_store': './torchserve_workspace/model_store',
                'initial_workers': 1,
                'max_workers': 4
            },
            'models': [
                {
                    'name': 'resnet18',
                    'source': 'torchvision',
                    'model_name': 'resnet18',
                    'pretrained': True,
                    'handler': 'image_classifier'
                }
            ]
        }
    
    def install_dependencies(self):
        """安装必要的依赖"""
        logger.info("检查并安装依赖...")
        
        dependencies = [
            'torch',
            'torchvision',
            'torchserve',
            'torch-model-archiver',
            'torch-workflow-archiver',
            'transformers',
            'Pillow',
            'requests',
            'pyyaml'
        ]
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                logger.info(f"✓ {dep} 已安装")
            except ImportError:
                logger.info(f"正在安装 {dep}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', dep], 
                             check=True, capture_output=True)
                logger.info(f"✓ {dep} 安装完成")
    
    def download_model(self, model_config: Dict) -> Tuple[str, str]:
        """
        下载模型
        
        Args:
            model_config: 模型配置
            
        Returns:
            (model_path, config_path): 模型文件路径和配置文件路径
        """
        model_name = model_config['name']
        source = model_config['source']
        
        logger.info(f"开始下载模型: {model_name} (源: {source})")
        
        if source == 'huggingface':
            return self._download_huggingface_model(model_config)
        elif source == 'torchvision':
            return self._download_torchvision_model(model_config)
        elif source == 'pytorch_hub':
            return self._download_pytorch_hub_model(model_config)
        elif source == 'local':
            return self._prepare_local_model(model_config)
        else:
            raise ValueError(f"不支持的模型源: {source}")
    
    def _download_huggingface_model(self, model_config: Dict) -> Tuple[str, str]:
        """下载HuggingFace模型"""
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        model_name = model_config['name']
        hf_model_name = model_config.get('hf_model_name', model_name)
        
        model_dir = self.temp_dir / f"{model_name}_hf"
        model_dir.mkdir(exist_ok=True)
        
        # 下载模型和tokenizer
        model = AutoModel.from_pretrained(hf_model_name)
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        config = AutoConfig.from_pretrained(hf_model_name)
        
        # 保存模型
        model_path = model_dir / "pytorch_model.bin"
        config_path = model_dir / "config.json"
        
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        return str(model_path), str(config_path)
    
    def _download_torchvision_model(self, model_config: Dict) -> Tuple[str, str]:
        """下载TorchVision模型"""
        import torch
        import torchvision.models as models
        
        model_name = model_config['name']
        torch_model_name = model_config.get('model_name', model_name)
        pretrained = model_config.get('pretrained', True)
        
        # 获取模型
        model_fn = getattr(models, torch_model_name)
        model = model_fn(pretrained=pretrained)
        model.eval()
        
        # 保存模型
        model_dir = self.temp_dir / f"{model_name}_tv"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
        
        # 保存模型状态
        torch.save(model.state_dict(), model_path)
        
        # 创建配置文件
        config = {
            "model_name": torch_model_name,
            "model_type": "torchvision",
            "pretrained": pretrained,
            "num_classes": getattr(model, 'num_classes', 1000)
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(model_path), str(config_path)
    
    def _download_pytorch_hub_model(self, model_config: Dict) -> Tuple[str, str]:
        """下载PyTorch Hub模型"""
        import torch
        
        model_name = model_config['name']
        repo = model_config['repo']
        model_hub_name = model_config['model_name']
        
        # 从PyTorch Hub加载模型
        model = torch.hub.load(repo, model_hub_name, pretrained=True)
        model.eval()
        
        # 保存模型
        model_dir = self.temp_dir / f"{model_name}_hub"
        model_dir.mkdir(exist_ok=True)
        
        model_path = model_dir / "model.pt"
        config_path = model_dir / "config.json"
        
        torch.save(model.state_dict(), model_path)
        
        config = {
            "model_name": model_hub_name,
            "model_type": "pytorch_hub",
            "repo": repo
        }
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(model_path), str(config_path)
    
    def _prepare_local_model(self, model_config: Dict) -> Tuple[str, str]:
        """准备本地模型"""
        model_path = model_config['model_path']
        config_path = model_config.get('config_path', '')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"本地模型文件不存在: {model_path}")
        
        return model_path, config_path
    
    def create_handler(self, model_config: Dict) -> str:
        """创建模型处理器"""
        handler_type = model_config.get('handler', 'default')
        model_name = model_config['name']
        
        if handler_type in ['image_classifier', 'text_classifier', 'object_detector']:
            # 使用内置处理器
            return handler_type
        
        # 创建自定义处理器
        handler_dir = self.temp_dir / f"{model_name}_handler"
        handler_dir.mkdir(exist_ok=True)
        
        handler_path = handler_dir / "handler.py"
        
        # 根据模型类型创建处理器
        if model_config['source'] == 'torchvision':
            handler_code = self._create_torchvision_handler(model_config)
        elif model_config['source'] == 'huggingface':
            handler_code = self._create_huggingface_handler(model_config)
        else:
            handler_code = self._create_default_handler(model_config)
        
        with open(handler_path, 'w') as f:
            f.write(handler_code)
        
        return str(handler_path)
    
    def _create_torchvision_handler(self, model_config: Dict) -> str:
        """创建TorchVision模型处理器"""
        return '''
import torch
import torchvision.transforms as transforms
from torchvision import models
from ts.torch_handler.image_classifier import ImageClassifier
import json

class TorchVisionHandler(ImageClassifier):
    """
    TorchVision 模型处理器
    """
    
    def __init__(self):
        super().__init__()
    
    def initialize(self, context):
        """初始化模型"""
        super().initialize(context)
        
        # 加载模型配置
        config_file = context.manifest.get("model").get("serializedFile")
        
        # 设置图像预处理
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
'''
    
    def _create_huggingface_handler(self, model_config: Dict) -> str:
        """创建HuggingFace模型处理器"""
        return '''
import torch
from transformers import AutoModel, AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler
import json

class HuggingFaceHandler(BaseHandler):
    """
    HuggingFace 模型处理器
    """
    
    def __init__(self):
        super().__init__()
        
    def initialize(self, context):
        """初始化模型和tokenizer"""
        super().initialize(context)
        
        # 从模型目录加载
        model_dir = context.system_properties.get("model_dir")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)
        self.model.eval()
        
    def preprocess(self, data):
        """预处理输入数据"""
        texts = []
        for row in data:
            if isinstance(row, dict):
                text = row.get("text", "")
            else:
                text = str(row)
            texts.append(text)
        
        # Tokenize
        inputs = self.tokenizer(texts, padding=True, truncation=True, 
                              return_tensors="pt", max_length=512)
        return inputs
        
    def inference(self, inputs):
        """模型推理"""
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs
        
    def postprocess(self, outputs):
        """后处理输出"""
        # 简单返回最后隐藏状态
        last_hidden_states = outputs.last_hidden_state
        return last_hidden_states.tolist()
'''
    
    def _create_default_handler(self, model_config: Dict) -> str:
        """创建默认处理器"""
        return '''
import torch
from ts.torch_handler.base_handler import BaseHandler

class DefaultHandler(BaseHandler):
    """
    默认模型处理器
    """
    
    def __init__(self):
        super().__init__()
        
    def initialize(self, context):
        """初始化模型"""
        super().initialize(context)
        
    def preprocess(self, data):
        """预处理输入数据"""
        return data
        
    def inference(self, data):
        """模型推理"""
        with torch.no_grad():
            outputs = self.model(data)
        return outputs
        
    def postprocess(self, outputs):
        """后处理输出"""
        return outputs.tolist()
'''
    
    def create_mar_file(self, model_config: Dict, model_path: str, handler_path: str) -> str:
        """创建MAR文件"""
        model_name = model_config['name']
        version = model_config.get('version', '1.0')
        
        mar_file = self.model_store / f"{model_name}.mar"
        
        # 构建torch-model-archiver命令
        cmd = [
            'torch-model-archiver',
            '--model-name', model_name,
            '--version', version,
            '--serialized-file', model_path,
            '--handler', handler_path,
            '--export-path', str(self.model_store)
        ]
        
        # 添加额外文件
        extra_files = model_config.get('extra_files', [])
        if extra_files:
            cmd.extend(['--extra-files', ','.join(extra_files)])
        
        # 添加requirements文件
        requirements = model_config.get('requirements')
        if requirements:
            cmd.extend(['--requirements-file', requirements])
        
        logger.info(f"创建MAR文件: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            logger.info(f"✓ MAR文件创建成功: {mar_file}")
            return str(mar_file)
        except subprocess.CalledProcessError as e:
            logger.error(f"创建MAR文件失败: {e.stderr}")
            raise
    
    def create_torchserve_config(self):
        """创建TorchServe配置文件"""
        config = self.config['torchserve']
        
        properties = [
            f"inference_address={config.get('inference_address', 'http://127.0.0.1:8080')}",
            f"management_address={config.get('management_address', 'http://127.0.0.1:8081')}",
            f"model_store={self.model_store}",
            f"initial_workers={config.get('initial_workers', 1)}",
            f"max_workers={config.get('max_workers', 4)}",
            "enable_envvars_config=true",
            "install_py_dep_per_model=true"
        ]
        
        with open(self.torchserve_config, 'w') as f:
            f.write('\n'.join(properties))
        
        logger.info(f"TorchServe配置文件已创建: {self.torchserve_config}")
    
    def start_torchserve(self):
        """启动TorchServe服务"""
        logger.info("启动TorchServe服务...")
        
        cmd = [
            'torchserve',
            '--start',
            '--ts-config', str(self.torchserve_config),
            '--ncs'  # 非阻塞模式
        ]
        
        try:
            self.torchserve_process = subprocess.Popen(
                cmd, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True
            )
            
            # 等待服务启动
            time.sleep(10)
            
            # 检查服务状态
            if self.check_torchserve_health():
                logger.info("✓ TorchServe服务启动成功")
            else:
                raise Exception("TorchServe服务启动失败")
                
        except Exception as e:
            logger.error(f"启动TorchServe失败: {e}")
            raise
    
    def stop_torchserve(self):
        """停止TorchServe服务"""
        logger.info("停止TorchServe服务...")
        
        try:
            subprocess.run(['torchserve', '--stop'], check=True)
            logger.info("✓ TorchServe服务已停止")
        except subprocess.CalledProcessError:
            logger.warning("停止TorchServe服务失败，尝试强制终止...")
            if self.torchserve_process:
                self.torchserve_process.terminate()
                self.torchserve_process.wait()
    
    def check_torchserve_health(self) -> bool:
        """检查TorchServe服务健康状态"""
        inference_url = self.config['torchserve']['inference_address']
        management_url = self.config['torchserve']['management_address']
        
        try:
            # 检查推理API
            response = requests.get(f"{inference_url}/ping", timeout=5)
            if response.status_code != 200:
                return False
            
            # 检查管理API
            response = requests.get(f"{management_url}/models", timeout=5)
            if response.status_code != 200:
                return False
            
            return True
        except requests.RequestException:
            return False
    
    def register_model(self, mar_file: str, model_config: Dict) -> bool:
        """注册模型到TorchServe"""
        model_name = model_config['name']
        management_url = self.config['torchserve']['management_address']
        
        register_url = f"{management_url}/models"
        
        params = {
            'url': mar_file,
            'initial_workers': model_config.get('initial_workers', 1),
            'synchronous': 'true'
        }
        
        try:
            response = requests.post(register_url, params=params, timeout=30)
            if response.status_code == 200:
                logger.info(f"✓ 模型 {model_name} 注册成功")
                return True
            else:
                logger.error(f"模型注册失败: {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"模型注册失败: {e}")
            return False
    
    def test_model(self, model_config: Dict) -> bool:
        """测试模型"""
        model_name = model_config['name']
        inference_url = self.config['torchserve']['inference_address']
        
        test_url = f"{inference_url}/predictions/{model_name}"
        
        # 根据模型类型准备测试数据
        test_data = self._prepare_test_data(model_config)
        
        try:
            if isinstance(test_data, dict):
                response = requests.post(test_url, json=test_data, timeout=30)
            else:
                response = requests.post(test_url, files=test_data, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✓ 模型 {model_name} 测试成功")
                logger.info(f"测试结果: {response.json()}")
                return True
            else:
                logger.error(f"模型测试失败: {response.text}")
                return False
        except requests.RequestException as e:
            logger.error(f"模型测试失败: {e}")
            return False
    
    def _prepare_test_data(self, model_config: Dict):
        """准备测试数据"""
        source = model_config['source']
        handler = model_config.get('handler', 'default')
        
        if handler == 'image_classifier' or source == 'torchvision':
            # 为图像分类模型创建测试图像
            from PIL import Image
            import numpy as np
            
            # 创建一个随机图像
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            img_path = self.temp_dir / "test_image.jpg"
            img.save(img_path)
            
            return {'data': open(img_path, 'rb')}
        
        elif source == 'huggingface':
            # 为文本模型创建测试文本
            return {"text": "This is a test sentence for the model."}
        
        else:
            # 默认测试数据
            return {"data": [[1, 2, 3, 4, 5]]}
    
    def deploy_models(self) -> bool:
        """部署所有模型"""
        logger.info("开始模型部署流程...")
        
        success_count = 0
        total_count = len(self.config['models'])
        
        for model_config in self.config['models']:
            try:
                model_name = model_config['name']
                logger.info(f"\n{'='*50}")
                logger.info(f"部署模型: {model_name}")
                logger.info(f"{'='*50}")
                
                # 1. 下载模型
                model_path, config_path = self.download_model(model_config)
                
                # 2. 创建处理器
                handler_path = self.create_handler(model_config)
                
                # 3. 创建MAR文件
                mar_file = self.create_mar_file(model_config, model_path, handler_path)
                
                # 4. 注册模型
                if self.register_model(mar_file, model_config):
                    # 5. 测试模型
                    if self.test_model(model_config):
                        success_count += 1
                        logger.info(f"✓ 模型 {model_name} 部署并测试成功")
                    else:
                        logger.error(f"✗ 模型 {model_name} 测试失败")
                else:
                    logger.error(f"✗ 模型 {model_name} 注册失败")
                    
            except Exception as e:
                logger.error(f"✗ 模型 {model_config['name']} 部署失败: {e}")
        
        logger.info(f"\n部署完成: {success_count}/{total_count} 个模型成功部署")
        return success_count == total_count
    
    def cleanup(self):
        """清理临时文件"""
        logger.info("清理临时文件...")
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
        logger.info("✓ 清理完成")
    
    def run(self):
        """运行完整的部署流程"""
        try:
            # 1. 安装依赖
            self.install_dependencies()
            
            # 2. 创建TorchServe配置
            self.create_torchserve_config()
            
            # 3. 启动TorchServe
            self.start_torchserve()
            
            # 4. 部署模型
            success = self.deploy_models()
            
            if success:
                logger.info("\n🎉 所有模型部署成功！")
                logger.info(f"推理API: {self.config['torchserve']['inference_address']}")
                logger.info(f"管理API: {self.config['torchserve']['management_address']}")
            else:
                logger.error("\n❌ 部分模型部署失败")
            
            return success
            
        except Exception as e:
            logger.error(f"部署过程中发生错误: {e}")
            return False
        finally:
            # 清理临时文件
            self.cleanup()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TorchServe 自动部署脚本')
    parser.add_argument('--config', '-c', default='deploy_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--stop', action='store_true',
                       help='停止TorchServe服务')
    parser.add_argument('--clean', action='store_true',
                       help='清理工作目录')
    
    args = parser.parse_args()
    
    deployer = TorchServeDeployer(args.config)
    
    if args.stop:
        deployer.stop_torchserve()
        return
    
    if args.clean:
        deployer.cleanup()
        return
    
    # 运行部署
    success = deployer.run()
    
    if not success:
        sys.exit(1)

if __name__ == '__main__':
    main()