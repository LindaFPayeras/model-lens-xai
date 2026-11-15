from torchvision.models import (
    resnet50,
    ResNet50_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights
)

from PIL import Image
from pathlib import Path

RESNET_WEIGHTS = ResNet50_Weights.DEFAULT
EFFICIENTNET_WEIGHTS = EfficientNet_B0_Weights.DEFAULT

resnet_transform = RESNET_WEIGHTS.transforms()
efficientnet_transform = EFFICIENTNET_WEIGHTS.transforms()

def preprocess_image(img_path: Path | str):
    original = Image.open(img_path).convert("RGB")

    # Aplicar transformaciones independientemente
    resnet_image = resnet_transform(original)
    efficientnet_image = efficientnet_transform(original)

    return original, resnet_image, efficientnet_image