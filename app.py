import gradio as gr
import torch
from style_transfer import StyleTransferPipeline
from PIL import Image
import numpy as np
import torch
# 初始化模型

pipeline = StyleTransferPipeline(
    decoder_path="checkpoints/decoder.pth",
    style_features_path="checkpoints/style_features.pt",
    encode_path="checkpoints/vgg_normalised.pth",
)

def tensor_to_pil(tensor):
    """
    tensor: [1,3,H,W] or [3,H,W], range [0,1]
    """
    if tensor.dim() == 4:
        tensor = tensor[0]

    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0)  # CHW → HWC
    # array = (tensor.cpu().numpy() * 256).astype(np.uint8)
    array = (tensor.cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)
def style_transfer_gradio(content_img, text_prompt, alpha_slider):
    alpha = alpha_slider
    
    result_img, style_path, similarity = pipeline.transfer(
        content_path=content_img,
        text_prompt=text_prompt,
        alpha=alpha
    )

    # Tensor → PIL
    result_img = tensor_to_pil(result_img)

    # 风格图路径 → PIL
    style_img = Image.open(style_path).convert("RGB")

    info = f"检索到的风格图: {style_path}\n相似度: {similarity:.4f}"

    return result_img, style_img, info

# 创建Gradio界面
demo = gr.Interface(
    fn=style_transfer_gradio,
    inputs=[
        gr.Image(type="pil", label="上传内容图"),
        gr.Textbox(label="风格描述", placeholder="例如: van gogh starry night"),
        gr.Slider(0, 1, value=1.0, label="风格强度")
    ],
    outputs=[
        gr.Image(label="风格化结果"),
        gr.Image(label="匹配的风格图"),
        gr.Textbox(label="检索信息")
    ],
    title="文本引导风格迁移 (CLIP + AdaIN)"
)

if __name__ == "__main__":
    demo.launch(share=False)
