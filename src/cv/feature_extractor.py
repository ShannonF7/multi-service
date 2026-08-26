import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os

# --- 1. 定义 SimCLR 模型结构 ---
class SimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(SimCLR, self).__init__()
        self.resnet_dict = {"resnet50": models.resnet50(pretrained=False, num_classes=out_dim)}
        self.backbone = self._get_basemodel(base_model)
        dim_mlp = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(dim_mlp, dim_mlp), 
            nn.ReLU(), 
            nn.Linear(dim_mlp, out_dim)
        )

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise ValueError(f"Invalid backbone architecture. Check the config file and pass one of: {self.resnet_dict.keys()}")
        return model

    def forward(self, x):
        return self.backbone(x)

# --- 2. 特征提取器封装类 ---
class FeatureExtractor:
    def __init__(self, model_path: str, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading SimCLR model from {model_path} on {self.device}...")
        
        # 初始化模型结构 (ResNet50, out_dim=128 是 SimCLR 的常用配置，需与训练时一致)
        # 如果您训练时 out_dim 不是 128，请修改这里
        self.model = SimCLR(base_model="resnet50", out_dim=128)
        
        # 加载权重
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
            
        checkpoint = torch.load(model_path, map_location=self.device)
        

        if 'model_state_dict' in checkpoint:
            print("Loading state_dict from checkpoint['model_state_dict']...")
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
       
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('encoder.'):
                new_key = k.replace('encoder.', 'backbone.')
                new_state_dict[new_key] = v
            elif k.startswith('projection.'):
                # projection.0.weight -> backbone.fc.0.weight
                # projection.0.bias -> backbone.fc.0.bias
                # projection.2.weight -> backbone.fc.2.weight
                # projection.2.bias -> backbone.fc.2.bias
                new_key = k.replace('projection.', 'backbone.fc.')
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
        
        # 再次尝试加载，允许非严格匹配 (strict=False) 以忽略 num_batches_tracked 等无关参数
        try:
            self.model.load_state_dict(new_state_dict, strict=False)
            print("✅ Successfully loaded model weights with key mapping.")
        except Exception as e:
            print(f"⚠️ Strict loading failed, trying original state_dict: {e}")
            self.model.load_state_dict(state_dict, strict=False)
            
        self.model.to(self.device)
        self.model.eval()

        # 定义预处理
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract(self, image_input) -> list:
        """
        输入图片路径或PIL Image对象，返回特征向量 (List[float])
        """
        try:
            if isinstance(image_input, str):
                image = Image.open(image_input).convert('RGB')
            else:
                image = image_input.convert('RGB')
                
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                feature = self.model(image)
                # 归一化特征向量 (SimCLR 检索通常需要归一化)
                feature = torch.nn.functional.normalize(feature, dim=1)
                
            # 转为 Python List 以便存入 pgvector
            return feature.cpu().numpy().flatten().tolist()
        except Exception as e:
            print(f"Error extracting feature: {e}")
            return []

# 全局单例
feature_extractor = None

def get_feature_extractor(model_path: str = None):
    global feature_extractor
    if feature_extractor is None:
        if model_path is None:
             # Default path if not provided
             base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
             model_path = os.path.join(base_dir, "models","simclr_checkpoints", "simclr_zhangbi_final.pth")
             
        feature_extractor = FeatureExtractor(model_path)
    return feature_extractor
