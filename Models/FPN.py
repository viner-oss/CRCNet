import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet50_FPN_Classification(nn.Module):
    def __init__(self, num_classes=3, pretrained=True):
        super(ResNet50_FPN_Classification, self).__init__()

        # -----------------------------------------------------------
        # 1. Backbone: ResNet50
        # -----------------------------------------------------------
        resnet = models.resnet50(pretrained=pretrained)

        old_conv1_weight = resnet.conv1.weight.data  # shape: [64, 3, 7, 7]
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data = old_conv1_weight.mean(dim=1, keepdim=True)  # shape: [64, 1, 7, 7]
        resnet.conv1 = new_conv1

        self.layer0 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1  # C2: stride 4, channels 256
        self.layer2 = resnet.layer2  # C3: stride 8, channels 512
        self.layer3 = resnet.layer3  # C4: stride 16, channels 1024
        self.layer4 = resnet.layer4  # C5: stride 32, channels 2048

        # -----------------------------------------------------------
        # 2. Neck: Feature Pyramid Network (FPN)
        # -----------------------------------------------------------
        fpn_dim = 256

        # Lateral connections (1x1 convs)
        self.lat_layer1 = nn.Conv2d(256, fpn_dim, kernel_size=1)  # For C2
        self.lat_layer2 = nn.Conv2d(512, fpn_dim, kernel_size=1)  # For C3
        self.lat_layer3 = nn.Conv2d(1024, fpn_dim, kernel_size=1)  # For C4
        self.lat_layer4 = nn.Conv2d(2048, fpn_dim, kernel_size=1)  # For C5

        # Smooth layers (3x3 convs)
        self.smooth1 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)  # For P2
        self.smooth2 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)  # For P3
        self.smooth3 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)  # For P4
        self.smooth4 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1)  # For P5

        # -----------------------------------------------------------
        # 3. Head: Classification
        # -----------------------------------------------------------

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Linear(fpn_dim * 4, num_classes)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='nearest') + y

    def forward(self, x):
        # --- Bottom-up Pathway (Backbone) ---
        # x shape: [B, 1, 224, 224]
        c1 = self.layer0(x)
        c2 = self.layer1(c1)  # 256 channels
        c3 = self.layer2(c2)  # 512 channels
        c4 = self.layer3(c3)  # 1024 channels
        c5 = self.layer4(c4)  # 2048 channels

        # --- Top-down Pathway (FPN) ---
        m5 = self.lat_layer4(c5)

        m4 = self._upsample_add(m5, self.lat_layer3(c4))

        m3 = self._upsample_add(m4, self.lat_layer2(c3))

        m2 = self._upsample_add(m3, self.lat_layer1(c2))

        # --- Smoothing ---
        p5 = self.smooth4(m5)
        p4 = self.smooth3(m4)
        p3 = self.smooth2(m3)
        p2 = self.smooth1(m2)

        # --- Classification Head (Fusion Strategy) ---
        v5 = self.avg_pool(p5).flatten(1)
        v4 = self.avg_pool(p4).flatten(1)
        v3 = self.avg_pool(p3).flatten(1)
        v2 = self.avg_pool(p2).flatten(1)

        out = torch.cat([v2, v3, v4, v5], dim=1)

        logits = self.fc(out)  # Shape: [B, 3]

        return logits



# -----------------------------------------------------------
if __name__ == "__main__":
    dummy_input = torch.randn(2, 1, 224, 224)

    model = ResNet50_FPN_Classification(num_classes=3)


    output = model(dummy_input)

    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")

