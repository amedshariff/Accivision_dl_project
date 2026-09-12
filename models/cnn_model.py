"""
CNN Spatial Feature Extractor Module (Person 3 - Level 5)
Leverages a pretrained ResNet18 backbone to extract 512-dim visual embeddings per frame.
"""
import torch
import torch.nn as nn
import torchvision.models as models

class CNNFeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, freeze_backbone: bool = True):
        super(CNNFeatureExtractor, self).__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        resnet = models.resnet18(weights=weights)
        
        # Remove the final classification FC layer
        # Output of avgpool is (B, 512, 1, 1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 512

        if freeze_backbone:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: (B, C, H, W)
        Output: (B, 512) feature embeddings
        """
        feats = self.feature_extractor(x)  # (B, 512, 1, 1)
        return torch.flatten(feats, 1)     # (B, 512)
