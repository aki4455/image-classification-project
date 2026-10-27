
from __future__ import annotations

import torch
from torch import nn
from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny


class TwoHeadConvNeXt(nn.Module):
    def __init__(self, n_brands: int, n_car_models: int, pretrained: bool = True):
        super().__init__()
        # pretrained=False at prediction time: no download, our trained weights are loaded afterwards
        net = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = net.features          # the pretrained backbone
        self.avgpool = net.avgpool            # (768, 7, 7) feature map -> (768, 1, 1)
        self.norm = net.classifier[0]         # ConvNeXt's own LayerNorm before its old output layer
        self.flatten = net.classifier[1]      # (768, 1, 1) -> 768 numbers
        dim = net.classifier[2].in_features   # 768
        self.brand_head = nn.Linear(dim, n_brands)
        self.car_model_head = nn.Linear(dim, n_car_models)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.flatten(self.norm(self.avgpool(self.features(x))))
        return self.brand_head(f), self.car_model_head(f)