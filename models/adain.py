import torch
import torch.nn as nn
import torch
import torch.nn as nn
class AdaIN(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, content_feat, style_feat):
        assert content_feat.size()[:2] == style_feat.size()[:2]

        B, C = content_feat.size()[:2]

        content_feat = content_feat.float()
        style_feat   = style_feat.float()

        content_mean = content_feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        content_std  = content_feat.view(B, C, -1).std(dim=2, unbiased=False).view(B, C, 1, 1)

        style_mean = style_feat.view(B, C, -1).mean(dim=2).view(B, C, 1, 1)
        style_std  = style_feat.view(B, C, -1).std(dim=2, unbiased=False).view(B, C, 1, 1)

        normalized = (content_feat - content_mean) / (content_std + self.eps)
        stylized   = normalized * style_std + style_mean

        return stylized


