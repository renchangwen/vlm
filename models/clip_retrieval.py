import torch
import clip
from PIL import Image
import json
from pathlib import Path
import torch.nn.functional as F
from typing import Dict, Tuple, List

class CLIPStyleRetriever:
    """基于CLIP的风格图检索器"""
    
    def __init__(self, style_dir: str, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
        # 加载CLIP模型
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()
        
        self.style_dir = Path(style_dir)
        self.style_features = {}
        
    def preprocess_style_library(self, style_labels_path: str, save_path: str = None):
        """预处理风格图库，计算CLIP特征"""
        
        # 加载风格图标签
        with open(style_labels_path, 'r', encoding='utf-8') as f:
            style_labels = json.load(f)
        
        print("正在预处理风格图库...")
        for img_name, text_label in style_labels.items():
            img_path = self.style_dir / img_name
            
            if not img_path.exists():
                print(f"警告: {img_path} 不存在")
                continue
            
            # 加载图像
            image = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)
            
            # 编码图像和文本
            with torch.no_grad():
                image_feature = self.model.encode_image(image)
                text_feature = self.model.encode_text(
                    clip.tokenize([text_label]).to(self.device)
                )
                
                # L2归一化
                image_feature = image_feature / image_feature.norm(dim=-1, keepdim=True)
                text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True)
            
            self.style_features[img_name] = {
                'image_feature': image_feature.cpu(),
                'text_feature': text_feature.cpu(),
                'text_label': text_label,
                'path': str(img_path)
            }
            
            print(f"  已处理: {img_name} - {text_label}")
        
        # 保存特征
        if save_path:
            torch.save(self.style_features, save_path)
            print(f"特征已保存到: {save_path}")
        
        return self.style_features
    def load_features(self, path: str):
        """加载预计算的风格特征"""
        self.style_features = torch.load(path)
        print(f"已加载风格特征: {path}")

    def retrieve(self, text_prompt: str, top_k: int = 1):
        """根据文本检索最相似的风格图"""
        text_tokens = clip.tokenize([text_prompt]).to(self.device)

        with torch.no_grad():
            text_feat = self.model.encode_text(text_tokens)
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        scores = []
        for name, data in self.style_features.items():
            img_feat = data["image_feature"].to(self.device)
            sim = F.cosine_similarity(text_feat, img_feat).item()
            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_style_path(self, style_name: str):
        """根据风格名返回图像路径"""
        return self.style_features[style_name]["path"]