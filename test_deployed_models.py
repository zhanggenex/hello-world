#!/usr/bin/env python3
"""
测试已部署的TorchServe模型
"""

import requests
import json
import time
import os
from PIL import Image
import numpy as np
import argparse
import yaml
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTester:
    """模型测试器"""
    
    def __init__(self, config_file="deploy_config.yaml"):
        """初始化测试器"""
        self.config = self._load_config(config_file)
        self.inference_url = self.config['torchserve']['inference_address']
        self.management_url = self.config['torchserve']['management_address']
        
    def _load_config(self, config_file):
        """加载配置文件"""
        with open(config_file, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def check_service_health(self):
        """检查TorchServe服务状态"""
        try:
            # 检查推理服务
            response = requests.get(f"{self.inference_url}/ping", timeout=5)
            if response.status_code != 200:
                logger.error("推理服务不可用")
                return False
            
            # 检查管理服务
            response = requests.get(f"{self.management_url}/models", timeout=5)
            if response.status_code != 200:
                logger.error("管理服务不可用")
                return False
            
            logger.info("✓ TorchServe服务运行正常")
            return True
            
        except requests.RequestException as e:
            logger.error(f"服务连接失败: {e}")
            return False
    
    def list_models(self):
        """列出所有已注册的模型"""
        try:
            response = requests.get(f"{self.management_url}/models")
            if response.status_code == 200:
                models = response.json()['models']
                logger.info(f"已注册的模型 ({len(models)} 个):")
                for model in models:
                    logger.info(f"  - {model['modelName']} (版本: {model['modelVersion']})")
                return models
            else:
                logger.error(f"获取模型列表失败: {response.text}")
                return []
        except requests.RequestException as e:
            logger.error(f"获取模型列表失败: {e}")
            return []
    
    def create_test_image(self, size=(224, 224)):
        """创建测试图像"""
        # 创建一个随机彩色图像
        img_array = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        
        # 保存临时图像文件
        temp_path = Path("test_image.jpg")
        img.save(temp_path)
        return temp_path
    
    def test_image_classifier(self, model_name):
        """测试图像分类模型"""
        logger.info(f"测试图像分类模型: {model_name}")
        
        # 创建测试图像
        img_path = self.create_test_image()
        
        try:
            url = f"{self.inference_url}/predictions/{model_name}"
            with open(img_path, 'rb') as f:
                files = {'data': f}
                response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ {model_name} 预测成功")
                logger.info(f"  预测结果: {result}")
                return True
            else:
                logger.error(f"✗ {model_name} 预测失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ {model_name} 测试异常: {e}")
            return False
        finally:
            # 清理临时文件
            if img_path.exists():
                img_path.unlink()
    
    def test_text_classifier(self, model_name):
        """测试文本分类模型"""
        logger.info(f"测试文本模型: {model_name}")
        
        test_texts = [
            "This is a positive review of the product.",
            "I hate this movie, it's terrible.",
            "The weather is nice today.",
            "Machine learning is fascinating."
        ]
        
        try:
            url = f"{self.inference_url}/predictions/{model_name}"
            
            for i, text in enumerate(test_texts):
                data = {"text": text}
                response = requests.post(url, json=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✓ {model_name} 文本 {i+1} 预测成功")
                    logger.info(f"  输入: {text}")
                    logger.info(f"  结果: {result}")
                else:
                    logger.error(f"✗ {model_name} 文本 {i+1} 预测失败: {response.text}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"✗ {model_name} 测试异常: {e}")
            return False
    
    def test_object_detector(self, model_name):
        """测试目标检测模型"""
        logger.info(f"测试目标检测模型: {model_name}")
        
        # 创建测试图像
        img_path = self.create_test_image((640, 640))
        
        try:
            url = f"{self.inference_url}/predictions/{model_name}"
            with open(img_path, 'rb') as f:
                files = {'data': f}
                response = requests.post(url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ {model_name} 检测成功")
                logger.info(f"  检测结果: {result}")
                return True
            else:
                logger.error(f"✗ {model_name} 检测失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ {model_name} 测试异常: {e}")
            return False
        finally:
            # 清理临时文件
            if img_path.exists():
                img_path.unlink()
    
    def test_generic_model(self, model_name):
        """测试通用模型"""
        logger.info(f"测试通用模型: {model_name}")
        
        try:
            url = f"{self.inference_url}/predictions/{model_name}"
            data = {"data": [[1, 2, 3, 4, 5]]}
            response = requests.post(url, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✓ {model_name} 预测成功")
                logger.info(f"  预测结果: {result}")
                return True
            else:
                logger.error(f"✗ {model_name} 预测失败: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"✗ {model_name} 测试异常: {e}")
            return False
    
    def test_model_performance(self, model_name, num_requests=10):
        """测试模型性能"""
        logger.info(f"测试模型性能: {model_name} ({num_requests} 次请求)")
        
        # 创建测试图像
        img_path = self.create_test_image()
        
        success_count = 0
        total_time = 0
        
        try:
            url = f"{self.inference_url}/predictions/{model_name}"
            
            for i in range(num_requests):
                start_time = time.time()
                
                with open(img_path, 'rb') as f:
                    files = {'data': f}
                    response = requests.post(url, files=files, timeout=30)
                
                end_time = time.time()
                request_time = end_time - start_time
                total_time += request_time
                
                if response.status_code == 200:
                    success_count += 1
                    logger.info(f"  请求 {i+1}: 成功 ({request_time:.2f}s)")
                else:
                    logger.error(f"  请求 {i+1}: 失败 - {response.text}")
            
            # 性能统计
            avg_time = total_time / num_requests
            success_rate = success_count / num_requests * 100
            
            logger.info(f"性能测试结果:")
            logger.info(f"  成功率: {success_rate:.1f}% ({success_count}/{num_requests})")
            logger.info(f"  平均响应时间: {avg_time:.2f}s")
            logger.info(f"  总时间: {total_time:.2f}s")
            
            return success_rate >= 90  # 90%以上成功率算通过
            
        except Exception as e:
            logger.error(f"性能测试异常: {e}")
            return False
        finally:
            if img_path.exists():
                img_path.unlink()
    
    def test_all_models(self, performance_test=False):
        """测试所有模型"""
        logger.info("开始测试所有已部署的模型...")
        
        # 检查服务状态
        if not self.check_service_health():
            logger.error("TorchServe服务不可用，退出测试")
            return False
        
        # 获取模型列表
        models = self.list_models()
        if not models:
            logger.error("没有找到已注册的模型")
            return False
        
        # 测试每个模型
        success_count = 0
        
        for model_config in self.config['models']:
            model_name = model_config['name']
            handler = model_config.get('handler', 'default')
            
            logger.info(f"\n{'='*50}")
            logger.info(f"测试模型: {model_name}")
            logger.info(f"{'='*50}")
            
            # 根据处理器类型选择测试方法
            if handler == 'image_classifier' or model_config['source'] == 'torchvision':
                success = self.test_image_classifier(model_name)
            elif handler == 'text_classifier' or model_config['source'] == 'huggingface':
                success = self.test_text_classifier(model_name)
            elif handler == 'object_detector':
                success = self.test_object_detector(model_name)
            else:
                success = self.test_generic_model(model_name)
            
            if success:
                success_count += 1
                
                # 性能测试
                if performance_test:
                    logger.info("开始性能测试...")
                    self.test_model_performance(model_name, 5)
            
            time.sleep(1)  # 短暂休息
        
        # 测试总结
        total_models = len(self.config['models'])
        logger.info(f"\n{'='*50}")
        logger.info(f"测试完成: {success_count}/{total_models} 个模型通过测试")
        logger.info(f"{'='*50}")
        
        return success_count == total_models
    
    def test_specific_model(self, model_name, performance_test=False):
        """测试特定模型"""
        logger.info(f"测试特定模型: {model_name}")
        
        # 检查服务状态
        if not self.check_service_health():
            return False
        
        # 查找模型配置
        model_config = None
        for config in self.config['models']:
            if config['name'] == model_name:
                model_config = config
                break
        
        if not model_config:
            logger.error(f"未找到模型配置: {model_name}")
            return False
        
        # 测试模型
        handler = model_config.get('handler', 'default')
        
        if handler == 'image_classifier' or model_config['source'] == 'torchvision':
            success = self.test_image_classifier(model_name)
        elif handler == 'text_classifier' or model_config['source'] == 'huggingface':
            success = self.test_text_classifier(model_name)
        elif handler == 'object_detector':
            success = self.test_object_detector(model_name)
        else:
            success = self.test_generic_model(model_name)
        
        # 性能测试
        if success and performance_test:
            logger.info("开始性能测试...")
            self.test_model_performance(model_name, 10)
        
        return success

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='测试TorchServe部署的模型')
    parser.add_argument('--config', '-c', default='deploy_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--model', '-m', 
                       help='测试特定模型（不指定则测试所有模型）')
    parser.add_argument('--performance', '-p', action='store_true',
                       help='进行性能测试')
    parser.add_argument('--list', '-l', action='store_true',
                       help='仅列出已注册的模型')
    
    args = parser.parse_args()
    
    tester = ModelTester(args.config)
    
    if args.list:
        tester.list_models()
        return
    
    if args.model:
        success = tester.test_specific_model(args.model, args.performance)
    else:
        success = tester.test_all_models(args.performance)
    
    if not success:
        exit(1)

if __name__ == '__main__':
    main()