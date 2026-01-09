import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=3, pretrained=True, dropout=0.5, freeze_backbone=False):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        
        if pretrained and freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print(f"[Model] ResNet18 Backbone has been FROZEN. Only head will be trained.")
          
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        return self.backbone(x)
