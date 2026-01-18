import torch.nn as nn
from torchvision.models import (
    efficientnet_b4,
    EfficientNet_B4_Weights,
)

def build_efficientnet_b4(
    num_classes: int,
    freeze_backbone: bool = True,
):
    """
    Builds an ImageNet-pretrained EfficientNet-B4
    with a custom classification head.
    """
    weights = EfficientNet_B4_Weights.IMAGENET1K_V1
    model = efficientnet_b4(weights=weights)

    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    if freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier"):
                param.requires_grad = False

    return model

def unfreeze_backbone(model):
    """
    Unfreezes all model parameters (for fine-tuning).
    """
    for param in model.parameters():
        param.requires_grad = True