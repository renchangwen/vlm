import torch
import torch.nn as nn

class VGGEncoder(nn.Module):
    """
    AdaIN 官方 VGG（完整），forward 时返回 relu4_1
    """
    def __init__(self, device='cuda'):
        super().__init__()

        self.vgg = nn.Sequential(
            nn.Conv2d(3, 3, 1),

            nn.ReflectionPad2d(1),
            nn.Conv2d(3, 64, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, 3),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, 3),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, 3),
            nn.ReLU(inplace=False),   # relu4_1

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(2, 2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=False),

            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 512, 3),
            nn.ReLU(inplace=False),
        )

        self.vgg.to(device)
        self.vgg.eval()

        for p in self.vgg.parameters():
            p.requires_grad = False

    def forward(self, x):
        # 只 forward 到 relu4_1
        for i in range(31):
            x = self.vgg[i](x)
        return x





