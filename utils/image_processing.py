import torch
from PIL import Image
import torchvision.transforms as transforms

def load_image(path, size=512):
    """加载图像并转为tensor"""
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])
    img = Image.open(path).convert('RGB')
    return transform(img).unsqueeze(0)

def tensor_to_pil(tensor):
    """Tensor转PIL"""
    img = tensor.cpu().squeeze(0).permute(1, 2, 0)
    img = (img.clamp(0, 1) * 255).numpy().astype('uint8')
    return Image.fromarray(img)

def save_image(tensor, path):
    """保存tensor为图像"""
    img = tensor_to_pil(tensor)
    img.save(path, quality=95)