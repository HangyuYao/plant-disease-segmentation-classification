from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image

from src.inference.config import ROOT, SEGMENTATION_SIZE


def conv_bn_relu(
    in_c: int,
    out_c: int,
    k: int = 3,
    s: int = 1,
    p: int = 1,
    d: int = 1,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, k, s, p, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(True),
    )


class ASPP(nn.Module):
    def __init__(
        self,
        in_c: int,
        out_c: int = 256,
        rates: tuple[int, ...] = (1, 6, 12, 18),
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [conv_bn_relu(in_c, out_c, k=1, p=0, d=1)]
            + [conv_bn_relu(in_c, out_c, k=3, p=rate, d=rate) for rate in rates[1:]]
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_c * len(self.branches), out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(torch.cat([branch(x) for branch in self.branches], dim=1))


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class UpBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int):
        super().__init__()
        self.conv1 = conv_bn_relu(in_c + skip_c, out_c)
        self.conv2 = conv_bn_relu(out_c, out_c)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


class TimmEncoder(nn.Module):
    def __init__(
        self,
        name: str = "efficientnet_b0",
        out_indices: tuple[int, ...] = (1, 2, 3, 4),
    ):
        super().__init__()
        self.enc = timm.create_model(
            name,
            features_only=True,
            pretrained=False,
            out_indices=out_indices,
        )
        self.channels = self.enc.feature_info.channels()

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        return self.enc(x)


class EfficientNetSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TimmEncoder("efficientnet_b0", (1, 2, 3, 4))
        c1, c2, c3, c4 = self.enc.channels
        self.aspp = ASPP(c4, 256)
        self.up3 = UpBlock(256, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = UpBlock(128, c1, 64)
        self.head = nn.Sequential(conv_bn_relu(64, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        c1, c2, c3, c4 = self.enc(x)
        x = self.aspp(c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        x = self.head(x)
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


class MobileNetSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TimmEncoder("mobilenetv2_100", (1, 2, 3, 4))
        c1, c2, c3, c4 = self.enc.channels
        self.lat4 = nn.Conv2d(c4, 256, 1)
        self.up3 = UpBlock(256, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = UpBlock(128, c1, 64)
        self.head = nn.Sequential(conv_bn_relu(64, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        c1, c2, c3, c4 = self.enc(x)
        x = self.lat4(c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        x = self.head(x)
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


class MCFFASeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = TimmEncoder("efficientnet_b0", (1, 2, 3, 4))
        c1, c2, c3, c4 = self.enc.channels
        self.multi = nn.Sequential(
            conv_bn_relu(c4, 256, d=1, p=1),
            conv_bn_relu(256, 256, d=2, p=2),
            conv_bn_relu(256, 256, d=4, p=4),
            SEBlock(256),
        )
        self.up3 = UpBlock(256, c3, 256)
        self.up2 = UpBlock(256, c2, 128)
        self.up1 = UpBlock(128, c1, 64)
        self.head = nn.Sequential(conv_bn_relu(64, 64), nn.Conv2d(64, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        height, width = x.shape[2], x.shape[3]
        c1, c2, c3, c4 = self.enc(x)
        x = self.multi(c4)
        x = self.up3(x, c3)
        x = self.up2(x, c2)
        x = self.up1(x, c1)
        x = self.head(x)
        return F.interpolate(x, size=(height, width), mode="bilinear", align_corners=False)


@dataclass(frozen=True)
class SegmentationModelSpec:
    label: str
    weight_path: Path
    builder: Callable[[], nn.Module]


SEGMENTATION_MODELS: dict[str, SegmentationModelSpec] = {
    "efficientnet": SegmentationModelSpec(
        label="EfficientNet lesion segmenter",
        weight_path=ROOT / "efficientnet_best.pt",
        builder=EfficientNetSeg,
    ),
    "mobilenet": SegmentationModelSpec(
        label="MobileNet lesion segmenter",
        weight_path=ROOT / "mobilenet_best.pt",
        builder=MobileNetSeg,
    ),
    "mcffa": SegmentationModelSpec(
        label="MCFFA lesion segmenter",
        weight_path=ROOT / "mcffa_best.pt",
        builder=MCFFASeg,
    ),
}


def build_segmenter(model_key: str) -> nn.Module:
    try:
        return SEGMENTATION_MODELS[model_key].builder()
    except KeyError as exc:
        raise ValueError(f"Unknown segmentation model: {model_key}") from exc


def load_segmenter(weight_path: Path, device: str, model_key: str = "efficientnet") -> nn.Module:
    model = build_segmenter(model_key).to(device)
    checkpoint = torch.load(weight_path, map_location=device)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def predict_mask(model: nn.Module, image: Image.Image, device: str) -> np.ndarray:
    original_size = image.size
    resized = image.convert("RGB").resize((SEGMENTATION_SIZE, SEGMENTATION_SIZE))
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    with torch.no_grad():
        mask = torch.sigmoid(model(tensor.unsqueeze(0).to(device)))[0, 0].cpu().numpy()
    mask_img = Image.fromarray((mask * 255).astype(np.uint8)).resize(
        original_size,
        Image.BILINEAR,
    )
    return np.array(mask_img).astype(np.float32) / 255.0
