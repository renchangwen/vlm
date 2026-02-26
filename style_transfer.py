from models.vgg import VGGEncoder
from models.adain import AdaIN
from models.decoder import Decoder
from models.clip_retrieval import CLIPStyleRetriever
import torch
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

class StyleTransferPipeline:
    """完整的文本引导风格迁移流程"""
    
    def __init__(self, decoder_path: str, style_features_path: str, encode_path: str,device: str = 'cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.encoder = VGGEncoder(device=self.device)
        state_dict = torch.load(
            encode_path,   # vgg_normalised.pth
            map_location=self.device
        )
        # 👇 这一行就是我说的「保持不变的本质」
        self.encoder.vgg.load_state_dict(state_dict)
        self.encoder.eval()
        for p in self.encoder.parameters():
            p.requires_grad = False

        self.adain = AdaIN()
        self.decoder = Decoder().to(self.device)

        if Path(decoder_path).exists():
            self.decoder.load_state_dict(
                torch.load(decoder_path, map_location=self.device)
            )
            print(f"已加载解码器: {decoder_path}")

        self.decoder.eval()
        
        # 初始化CLIP检索器
        self.retriever = CLIPStyleRetriever(
            style_dir="data/styles",
            device=self.device
        )
        self.retriever.load_features(style_features_path)
        
    def vgg_normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1,3,1,1)
        std  = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1,3,1,1)
        return (x - mean) / std

    def transfer(self, content_path: str, text_prompt: str, alpha: float = 0.2):
        """执行风格迁移"""
        print(type(content_path))
        # 1. CLIP 检索风格图
        results = self.retriever.retrieve(text_prompt, top_k=1)
        style_name, similarity = results[0]
        style_path = self.retriever.get_style_path(style_name)

        # 2. 图像预处理
        transform = T.Compose([
            T.Resize((512, 512)),
            T.ToTensor()
        ])

        # content_img = transform(
        #     Image.open(content_path).convert("RGB")
        # ).unsqueeze(0).to(self.device)
        if isinstance(content_path, Image.Image):
            content_pil = content_path.convert("RGB")
        else:
            content_pil = Image.open(content_path).convert("RGB")

        content_img = transform(content_pil).unsqueeze(0).to(self.device)

        style_img = transform(
            Image.open(style_path).convert("RGB")
        ).unsqueeze(0).to(self.device)

        # 3. 提取特征
        with torch.no_grad():
            content_img = self.vgg_normalize(content_img)
            style_img   = self.vgg_normalize(style_img)
            content_feat = self.encoder(content_img)
            style_feat = self.encoder(style_img)

            # 4. AdaIN
            stylized_feat = self.adain(content_feat, style_feat)

            # 5. 风格强度控制
            stylized_feat = alpha * stylized_feat + (1 - alpha) * content_feat

            # 6. 解码
            result = self.decoder(stylized_feat)

            # # ⭐⭐⭐ 关键修复：强制输出合法图像范围
            # result = torch.clamp(result, 0, 1)

        return result, style_path, similarity



