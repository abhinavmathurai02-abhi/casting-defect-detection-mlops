"""model.py — Stage 2: transfer-learning model + embedding extractor.

Configure a pretrained ResNet18 backbone for transfer learning (freeze the backbone,
replace the final layer with a fresh 2-class head). The same backbone is reused as a
512-dim feature extractor for embedding drift. See notebook "Model Development".
"""
from __future__ import annotations

from pathlib import Path
import torch
import torch.nn as nn

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from torchvision.models import resnet18, ResNet18_Weights


from torchvision.models import resnet18, ResNet18_Weights

def build_model(freeze: bool | None = None) -> nn.Module:
    """
    Build an ImageNet-pretrained ResNet18 for binary classification.

    Parameters
    ----------
    freeze : bool | None, optional
        Whether to freeze the pretrained backbone.
        If None, uses config.FREEZE_BACKBONE.

    Returns
    -------
    nn.Module
        Configured ResNet18 model.
    """

    if freeze is None:
        freeze = config.FREEZE_BACKBONE

    # Load pretrained ResNet18
    net = resnet18(weights=ResNet18_Weights.DEFAULT)

    # Freeze backbone parameters
    if freeze:
        for param in net.parameters():
            param.requires_grad = False

    # Replace the final classification layer
    in_features = net.fc.in_features
    net.fc = nn.Linear(
        in_features=in_features,
        out_features=config.NUM_CLASSES,
    )

    # Ensure the new head is trainable
    for param in net.fc.parameters():
        param.requires_grad = True

    # Print model statistics
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(
        p.numel() for p in net.parameters()
        if p.requires_grad
    )

    print(f"Total Parameters     : {total_params:,}")
    print(f"Trainable Parameters : {trainable_params:,}")
    print(f"Frozen Parameters     : {total_params - trainable_params:,}")

    return net


def trainable_parameters(net: nn.Module):
    return [p for p in net.parameters() if p.requires_grad]


class EmbeddingExtractor(nn.Module):
    """
    Expose the 512-dimensional penultimate embedding
    from ResNet18 (everything except the final FC layer).
    """

    def __init__(self, net: nn.Module):
        super().__init__()

        # Keep every layer except the final classifier
        self.backbone = nn.Sequential(
            *list(net.children())[:-1]
        )

    @torch.no_grad()
    def forward(self, x):

        features = self.backbone(x)

        # Shape:
        # (batch_size, 512, 1, 1)
        features = torch.flatten(
            features,
            start_dim=1,
        )

        # Shape:
        # (batch_size, 512)
        return features



def save_model(net: nn.Module, path: Path | None = None) -> None:
    torch.save(net.state_dict(), path or config.MODEL_PATH)


def load_model(path: Path | None = None, freeze: bool = True) -> nn.Module:
    net = build_model(freeze=freeze)
    net.load_state_dict(torch.load(path or config.MODEL_PATH, map_location=config.DEVICE))
    net.eval()
    return net
