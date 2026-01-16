from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Optional, List, Dict
import matplotlib.pyplot as plt 
import numpy as np

from captum.attr import (
    Saliency,
    LayerGradCam
)

from captum.attr import visualization as viz


# ==========================================================
# Base XAI Inspector (Strategy pattern)
# ==========================================================
class XAIBase(ABC):
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
    ):
        self.model = model.eval()
        self.target_layer = target_layer
        self.device = device or next(model.parameters()).device

    @abstractmethod
    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[int] = None,
    ) -> torch.Tensor:
        pass


# ==========================================================
# Saliency Maps
# ==========================================================
class SaliencyXAI(XAIBase):
    def __init__(self, model: nn.Module, device=None):
        super().__init__(model, device=device)
        self.method = Saliency(self.model)

    def attribute(self, x: torch.Tensor, target: Optional[int] = None):
        x = x.requires_grad_()
        return self.method.attribute(x, target=target)


# ==========================================================
# Grad-CAM
# ==========================================================
class GradCAMXAI(XAIBase):
    def __init__(
        self,
        model: nn.Module,
        target_layer,
        device=None,
    ):
        super().__init__(model, target_layer, device)

        # IMPORTANT: forward_func must be callable
        self.method = LayerGradCam(
            forward_func=self.model.forward,
            layer=self.target_layer,
        )

    def attribute(
        self,
        x: torch.Tensor,
        target: Optional[int] = None,
    ):
        # Captum returns attribution at layer resolution
        attr = self.method.attribute(x, target=target)

        # Upsample to input resolution
        return F.interpolate(
            attr,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )



# ==========================================================
# Unified Inspector Facade
# ==========================================================
class ModelInspector:
    """
    High-level API for inspecting CNN-based classifiers.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        device: Optional[torch.device] = None,
    ):
        self.device = device or next(model.parameters()).device
        self.model = model.to(self.device).eval()

        self.methods: Dict[str, XAIBase] = {
            "saliency": SaliencyXAI(self.model, self.device),
            "gradcam": GradCAMXAI(self.model, target_layer, self.device)
        }

    @torch.no_grad()
    def predict(self, x: torch.Tensor):
        logits = self.model(x)
        return torch.argmax(logits, dim=1), logits.softmax(dim=1)

    def explain(
        self,
        x: torch.Tensor,
        method: str,
        target: Optional[int] = None,
    ) -> torch.Tensor:
        if method not in self.methods:
            raise ValueError(f"Unknown XAI method: {method}")

        x = x.to(self.device)
        return self.methods[method].attribute(x, target=target)

    def visualize_side_by_side(
        self,
        attribution: torch.Tensor,
        image: torch.Tensor,
        title: str = "",
        figsize: tuple[int,int] = (8,4)
    ):
        """
        Plots the input image and the attribution heatmap side by side.
        """
        # Ensure CPU numpy arrays
        attr = attribution.detach().cpu().numpy()
        img = image.detach().cpu().numpy()

        # Convert CHW -> HWC
        img_hwc = img.transpose(1, 2, 0)
        attr_hwc = attr.transpose(1, 2, 0)

        # Create figure
        fig, axs = plt.subplots(1, 2, figsize=figsize)

        # Left: original image
        axs[0].imshow(img_hwc)
        axs[0].set_title("Input Image")
        axs[0].axis("off")

        # Right: attribution heatmap
        viz.visualize_image_attr(
            attr_hwc,
            img_hwc,
            method="heat_map",
            sign="absolute_value",
            show_colorbar=True,
            title=title,
            plt_fig_axis=(fig, axs[1])
        )

        plt.tight_layout()
        plt.show()

    def visualize_overlay(
        self,
        attribution: torch.Tensor,
        image: torch.Tensor,
        title: str = "",
        alpha: float = 0.5,
        cmap: str = "jet",
        figsize: tuple[int,int] = (5,5)
    ):
        """
        Overlay attribution heatmap on top of input image with true alpha blending.

        Args:
            attribution: torch.Tensor, CHW format
            image: torch.Tensor, CHW format, same shape as attribution
            title: str
            alpha: float (0-1), transparency of heatmap
            cmap: matplotlib colormap
        """
        # Convert to numpy
        attr = attribution.detach().cpu().numpy()
        img = image.detach().cpu().numpy()

        # CHW -> HWC
        img_hwc = img.transpose(1,2,0)
        attr_hwc = attr.transpose(1,2,0)

        # Normalize attribution to 0-1
        attr_norm = np.max(attr_hwc, axis=2)
        attr_norm = (attr_norm - attr_norm.min()) / (attr_norm.max() - attr_norm.min() + 1e-8)

        # Plot
        fig, ax = plt.subplots(1,1,figsize=figsize)
        ax.imshow(img_hwc)  # original image
        ax.imshow(attr_norm, cmap=cmap, alpha=alpha)  # heatmap overlay
        ax.axis("off")
        ax.set_title(title)
        plt.tight_layout()
        plt.show()