"""
OSNet: Omni-Scale Feature Learning for Person Re-Identification.
Implementation of OSNet architecture with multi-stream convolutional residual blocks
and unified aggregation gates (Zhou et al., ICCV 2019).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.reid.base_extractor import BaseReIDExtractor, normalize_features


class Conv1x1(nn.Module):
    """1x1 convolution + BatchNorm + ReLU"""
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv3x3(nn.Module):
    """3x3 convolution + BatchNorm + ReLU"""
    def __init__(self, in_planes, out_planes, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightConv3x3(nn.Module):
    """Lightweight 3x3 depthwise-separable conv: 1x1 conv followed by 3x3 depthwise conv"""
    def __init__(self, in_planes, out_planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, 1, bias=False)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, 1, 1, bias=False, groups=out_planes)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv2(self.conv1(x))))


class AggregationGate(nn.Module):
    """Unified Aggregation Gate dynamically weights multi-scale streams."""
    def __init__(self, in_planes):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_planes, in_planes // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_planes // 4, in_planes, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        w = self.global_pool(x).view(b, c)
        w = self.fc(w).view(b, c, 1, 1)
        return x * w


class OSBlock(nn.Module):
    """Omni-Scale Residual Block with 4 scale streams and aggregation gate."""
    def __init__(self, in_planes, out_planes, bottleneck_reduction=4):
        super().__init__()
        mid_planes = out_planes // bottleneck_reduction
        self.conv1 = Conv1x1(in_planes, mid_planes)

        # Stream 1: Scale 1 (1x LightConv)
        self.conv2a = LightConv3x3(mid_planes, mid_planes)
        # Stream 2: Scale 2 (2x LightConv)
        self.conv2b = nn.Sequential(
            LightConv3x3(mid_planes, mid_planes),
            LightConv3x3(mid_planes, mid_planes)
        )
        # Stream 3: Scale 3 (3x LightConv)
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_planes, mid_planes),
            LightConv3x3(mid_planes, mid_planes),
            LightConv3x3(mid_planes, mid_planes)
        )
        # Stream 4: Scale 4 (4x LightConv)
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_planes, mid_planes),
            LightConv3x3(mid_planes, mid_planes),
            LightConv3x3(mid_planes, mid_planes),
            LightConv3x3(mid_planes, mid_planes)
        )

        self.gate = AggregationGate(mid_planes)
        self.conv3 = Conv1x1(mid_planes, out_planes)
        self.downsample = None
        if in_planes != out_planes:
            self.downsample = Conv1x1(in_planes, out_planes)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x1 = self.conv1(x)

        s1 = self.conv2a(x1)
        s2 = self.conv2b(x1)
        s3 = self.conv2c(x1)
        s4 = self.conv2d(x1)

        # Multi-scale aggregation
        out = self.gate(s1 + s2 + s3 + s4)
        out = self.conv3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        return self.relu(out)


class OSNet(BaseReIDExtractor):
    """
    OSNet model for Person Re-Identification.
    Produces a 512-dimensional normalized embedding vector.
    """
    def __init__(self, embedding_dim: int = 512, num_classes: int = 751, normalize: bool = True):
        super().__init__(embedding_dim=embedding_dim, normalize=normalize)
        self.num_classes = num_classes

        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # OSNet stages
        self.stage1 = nn.Sequential(
            OSBlock(64, 256),
            OSBlock(256, 256)
        )
        self.transition1 = nn.Sequential(
            Conv1x1(256, 256),
            nn.AvgPool2d(2, stride=2)
        )

        self.stage2 = nn.Sequential(
            OSBlock(256, 384),
            OSBlock(384, 384)
        )
        self.transition2 = nn.Sequential(
            Conv1x1(384, 384),
            nn.AvgPool2d(2, stride=2)
        )

        self.stage3 = nn.Sequential(
            OSBlock(384, 512),
            OSBlock(512, 512)
        )

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(512, embedding_dim, bias=False),
            nn.BatchNorm1d(embedding_dim)
        )

        # Classification head for training (BNNeck)
        self.classifier = nn.Linear(embedding_dim, num_classes, bias=False) if num_classes > 0 else None

    def forward_features(self, x: torch.Tensor, cam_ids: torch.Tensor = None) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stage1(x)
        x = self.transition1(x)
        x = self.stage2(x)
        x = self.transition2(x)
        x = self.stage3(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        feat = self.fc(x)
        return feat

    def forward_train(self, x: torch.Tensor, cam_ids: torch.Tensor = None):
        feat = self.forward_features(x, cam_ids)
        norm_feat = normalize_features(feat)
        cls_score = self.classifier(feat) if self.classifier is not None else None
        return norm_feat, cls_score
