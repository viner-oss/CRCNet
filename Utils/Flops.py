import torch
from fvcore.nn import FlopCountAnalysis
import Models.MobileNetV1

model = Models.MobileNetV1.MobileNetV1().eval()
img = torch.randn(size=[1, 1, 224, 224], dtype=torch.float32)

flops = FlopCountAnalysis(model, (img, ))
print('Flops:', flops.total())
print(flops.by_operator())