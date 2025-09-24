#!/usr/bin/env python3
"""
TorchServe 使用示例
演示如何调用已部署的模型进行推理
"""

import requests
import json
import base64
from PIL import Image
import numpy as np
from pathlib import Path
import argparse

class TorchServeClient:
    """TorchServe 客户端"""
    
    def __init__(self, base_url="http://127.0.0.1:8080"):
        """
        初始化客户端
        
        Args:
            base_url: TorchServe推理服务地址
        """
        self.base_url = base_url
    
    def predict_image(self, model_name, image_path):
        """
        图像预测
        
        Args:
            model_name: 模型名称
            image_path: 图像路径
            
        Returns:
            预测结果
        """
        url = f"{self.base_url}/predictions/{model_name}"
        
        with open(image_path, 'rb') as f:
            files = {'data': f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"预测失败: {response.text}")
    
    def predict_text(self, model_name, text):
        """
        文本预测
        
        Args:
            model_name: 模型名称
            text: 输入文本
            
        Returns:
            预测结果
        """
        url = f"{self.base_url}/predictions/{model_name}"
        
        data = {"text": text}
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"预测失败: {response.text}")
    
    def predict_batch(self, model_name, data_list):
        """
        批量预测
        
        Args:
            model_name: 模型名称
            data_list: 数据列表
            
        Returns:
            预测结果列表
        """
        results = []
        for data in data_list:
            if isinstance(data, str) and Path(data).exists():
                # 图像文件
                result = self.predict_image(model_name, data)
            else:
                # 文本或其他数据
                result = self.predict_text(model_name, data)
            results.append(result)
        
        return results
    
    def get_model_info(self, management_url="http://127.0.0.1:8081"):
        """
        获取模型信息
        
        Args:
            management_url: 管理API地址
            
        Returns:
            模型信息列表
        """
        response = requests.get(f"{management_url}/models")
        if response.status_code == 200:
            return response.json()['models']
        else:
            raise Exception(f"获取模型信息失败: {response.text}")

def create_sample_image(save_path="sample_image.jpg", size=(224, 224)):
    """创建示例图像"""
    # 创建一个彩色渐变图像
    width, height = size
    img_array = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            img_array[i, j] = [
                int(255 * i / height),      # 红色渐变
                int(255 * j / width),       # 绿色渐变
                int(255 * (i + j) / (height + width))  # 蓝色渐变
            ]
    
    img = Image.fromarray(img_array)
    img.save(save_path)
    print(f"示例图像已创建: {save_path}")
    return save_path

def demo_image_classification():
    """演示图像分类"""
    print("=== 图像分类演示 ===")
    
    client = TorchServeClient()
    
    # 创建示例图像
    image_path = create_sample_image()
    
    try:
        # 预测
        result = client.predict_image("resnet18", image_path)
        print(f"图像分类结果: {result}")
        
        # 解析结果（假设返回的是类别概率）
        if isinstance(result, list) and len(result) > 0:
            top_predictions = sorted(enumerate(result), key=lambda x: x[1], reverse=True)[:5]
            print("Top 5 预测:")
            for idx, (class_idx, prob) in enumerate(top_predictions):
                print(f"  {idx+1}. 类别 {class_idx}: {prob:.4f}")
    
    except Exception as e:
        print(f"图像分类失败: {e}")
    
    finally:
        # 清理示例图像
        if Path(image_path).exists():
            Path(image_path).unlink()

def demo_text_processing():
    """演示文本处理"""
    print("\n=== 文本处理演示 ===")
    
    client = TorchServeClient()
    
    test_texts = [
        "This is a wonderful day!",
        "I love machine learning.",
        "The weather is terrible today.",
        "Artificial intelligence is the future."
    ]
    
    try:
        for text in test_texts:
            result = client.predict_text("bert_base", text)
            print(f"输入: {text}")
            print(f"输出: {result}")
            print("-" * 50)
    
    except Exception as e:
        print(f"文本处理失败: {e}")

def demo_batch_prediction():
    """演示批量预测"""
    print("\n=== 批量预测演示 ===")
    
    client = TorchServeClient()
    
    # 创建多个示例图像
    image_paths = []
    for i in range(3):
        path = f"sample_image_{i}.jpg"
        create_sample_image(path, size=(224, 224))
        image_paths.append(path)
    
    try:
        # 批量预测
        results = client.predict_batch("resnet18", image_paths)
        
        for i, result in enumerate(results):
            print(f"图像 {i+1} 预测结果: {result}")
    
    except Exception as e:
        print(f"批量预测失败: {e}")
    
    finally:
        # 清理示例图像
        for path in image_paths:
            if Path(path).exists():
                Path(path).unlink()

def demo_model_info():
    """演示获取模型信息"""
    print("\n=== 模型信息演示 ===")
    
    client = TorchServeClient()
    
    try:
        models = client.get_model_info()
        print(f"已部署的模型 ({len(models)} 个):")
        
        for model in models:
            print(f"- 模型名称: {model['modelName']}")
            print(f"  版本: {model['modelVersion']}")
            print(f"  状态: {model.get('loadedAtStartup', 'Unknown')}")
            print(f"  Workers: {model.get('minWorkers', 0)} - {model.get('maxWorkers', 0)}")
            print("-" * 30)
    
    except Exception as e:
        print(f"获取模型信息失败: {e}")

def performance_test(model_name="resnet18", num_requests=10):
    """性能测试"""
    print(f"\n=== 性能测试演示 ({model_name}) ===")
    
    client = TorchServeClient()
    
    # 创建测试图像
    image_path = create_sample_image("perf_test_image.jpg")
    
    import time
    
    try:
        success_count = 0
        total_time = 0
        
        print(f"开始 {num_requests} 次请求测试...")
        
        for i in range(num_requests):
            start_time = time.time()
            
            try:
                result = client.predict_image(model_name, image_path)
                success_count += 1
                print(f"✓ 请求 {i+1}: 成功")
            except Exception as e:
                print(f"✗ 请求 {i+1}: 失败 - {e}")
            
            end_time = time.time()
            total_time += (end_time - start_time)
        
        # 统计结果
        avg_time = total_time / num_requests
        success_rate = success_count / num_requests * 100
        
        print(f"\n性能测试结果:")
        print(f"成功率: {success_rate:.1f}% ({success_count}/{num_requests})")
        print(f"平均响应时间: {avg_time:.3f}秒")
        print(f"总耗时: {total_time:.3f}秒")
        print(f"QPS: {num_requests / total_time:.2f} 请求/秒")
    
    except Exception as e:
        print(f"性能测试失败: {e}")
    
    finally:
        if Path(image_path).exists():
            Path(image_path).unlink()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='TorchServe使用示例')
    parser.add_argument('--demo', choices=['image', 'text', 'batch', 'info', 'perf', 'all'],
                       default='all', help='选择演示类型')
    parser.add_argument('--model', default='resnet18', help='模型名称')
    parser.add_argument('--url', default='http://127.0.0.1:8080', help='TorchServe URL')
    parser.add_argument('--requests', type=int, default=10, help='性能测试请求数')
    
    args = parser.parse_args()
    
    print("TorchServe 使用示例")
    print("=" * 50)
    
    if args.demo == 'image' or args.demo == 'all':
        demo_image_classification()
    
    if args.demo == 'text' or args.demo == 'all':
        demo_text_processing()
    
    if args.demo == 'batch' or args.demo == 'all':
        demo_batch_prediction()
    
    if args.demo == 'info' or args.demo == 'all':
        demo_model_info()
    
    if args.demo == 'perf' or args.demo == 'all':
        performance_test(args.model, args.requests)
    
    print("\n演示完成！")

if __name__ == '__main__':
    main()