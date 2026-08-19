"""
CRAFT text detector, own implementation instead of the craft-text-detector
pip package (that package pins numpy==1.21.2 and doesn't install on
anything recent). architecture below is checked against the real
craft_mlt_25k.pth checkpoint
"""

import cv2
import numpy as np
import pytesseract
import torch
import torch.nn as nn
import torchvision.models as tv_models


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, mid_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=1),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class VGG16FeatureExtractor(nn.Module):
    # vgg16-bn backbone, sliced where CRAFT actually slices it.
    # slice5 is NOT part of vgg's own layers - it's a fresh
    # maxpool + dilated conv + 1x1 conv block bolted on after slice4,
    # matches the checkpoint's "basenet.slice5.*" keys.
    def __init__(self, pretrained: bool = False):
        super().__init__()
        vgg = tv_models.vgg16_bn(weights="DEFAULT" if pretrained else None)
        features = vgg.features

        self.slice1 = features[:12]
        self.slice2 = features[12:19]
        self.slice3 = features[19:29]
        self.slice4 = features[29:39]

        self.slice5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
            nn.Conv2d(1024, 1024, kernel_size=1),
        )

    def forward(self, x: torch.Tensor):
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        h5 = self.slice5(h4)
        return h1, h2, h3, h4, h5


class CRAFTNet(nn.Module):
    # u-net style decoder over the vgg features. output is a 2-channel
    # map: region score (per-char likelihood) and affinity score (how
    # linked adjacent chars are, used to group chars into words).
    def __init__(self, pretrained_backbone: bool = False):
        super().__init__()
        self.basenet = VGG16FeatureExtractor(pretrained=pretrained_backbone)

        self.upconv1 = DoubleConv(1024 + 512, 512, 256)
        self.upconv2 = DoubleConv(256 + 512, 256, 128)
        self.upconv3 = DoubleConv(128 + 256, 128, 64)
        self.upconv4 = DoubleConv(64 + 128, 64, 32)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1, h2, h3, h4, h5 = self.basenet(x)

        y = torch.cat([h5, h4], dim=1)
        y = self.upconv1(y)
        y = nn.functional.interpolate(y, size=h3.shape[2:], mode="bilinear", align_corners=False)

        y = torch.cat([y, h3], dim=1)
        y = self.upconv2(y)
        y = nn.functional.interpolate(y, size=h2.shape[2:], mode="bilinear", align_corners=False)

        y = torch.cat([y, h2], dim=1)
        y = self.upconv3(y)
        y = nn.functional.interpolate(y, size=h1.shape[2:], mode="bilinear", align_corners=False)

        y = torch.cat([y, h1], dim=1)
        y = self.upconv4(y)

        return self.conv_cls(y)


_model = None


def load_model(weights_path: str, device: str = "cpu", verbose: bool = True) -> CRAFTNet:
    global _model
    if _model is not None:
        return _model

    model = CRAFTNet()
    state_dict = torch.load(weights_path, map_location=device)
    # official checkpoint keys are wrapped under "module." from
    # DataParallel training, strip that prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    result = model.load_state_dict(state_dict, strict=False)
    if verbose:
        print(f"CRAFT weight load - missing: {len(result.missing_keys)}, "
              f"unexpected: {len(result.unexpected_keys)}")
        if result.missing_keys:
            print("missing:", result.missing_keys[:10])
        if result.unexpected_keys:
            print("unexpected:", result.unexpected_keys[:10])

    model.eval()
    _model = model
    return model


def preprocess(img: np.ndarray, target_size: int = 1280) -> tuple[torch.Tensor, float]:
    h, w = img.shape[:2]
    scale = target_size / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    normalized = (rgb - mean) / std

    tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
    return tensor, scale


def boxes_from_score_map(
    region_score: np.ndarray,
    text_threshold: float = 0.7,
    low_threshold: float = 0.4,
) -> list[tuple[int, int, int, int]]:
    binary = (region_score > low_threshold).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)

    boxes = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 10:
            continue

        component_scores = region_score[labels == i]
        if component_scores.max() < text_threshold:
            continue

        boxes.append((x, y, w, h))

    return boxes


def detect_boxes(
    img: np.ndarray,
    weights_path: str,
    device: str = "cpu",
) -> list[tuple[int, int, int, int]]:
    model = load_model(weights_path, device=device)
    tensor, scale = preprocess(img)

    with torch.no_grad():
        out = model(tensor.to(device))

    region_score = out[0, 0].cpu().numpy()
    boxes = boxes_from_score_map(region_score)

    # score map comes out at half res on the resized image, scale
    # boxes back to the original image's coordinate space
    rescaled = []
    for x, y, w, h in boxes:
        rescaled.append((
            int(x * 2 / scale),
            int(y * 2 / scale),
            int(w * 2 / scale),
            int(h * 2 / scale),
        ))

    return rescaled


def recognize(
    img: np.ndarray,
    weights_path: str,
    device: str = "cpu",
    min_conf: float = 0.0,
) -> list[tuple[str, int, int, int, int, float]]:
    # detects boxes with CRAFT, reads text in each with tesseract -
    # counts as an independent ensemble vote since the region proposals
    # differ from what tesseract finds on its own
    boxes = detect_boxes(img, weights_path, device=device)

    results = []
    for x, y, w, h in boxes:
        crop = img[max(0, y) : y + h, max(0, x) : x + w]
        if crop.size == 0:
            continue

        text = pytesseract.image_to_string(crop).strip()
        if not text:
            continue

        data = pytesseract.image_to_data(crop, output_type=pytesseract.Output.DICT)
        confs = [float(c) for c in data["conf"] if float(c) >= 0]
        conf = (sum(confs) / len(confs) / 100.0) if confs else 0.5

        if conf < min_conf:
            continue

        results.append((text, x, y, w, h, conf))

    return results
