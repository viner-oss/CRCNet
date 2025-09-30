import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model.eval()
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, outp):
            self.activations = outp.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def __call__(self, input_tensor, class_idx=None):
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[:, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)

        grads = self.gradients      # (1, C, h, w)
        acts  = self.activations    # (1, C, h, w)
        weights = grads.mean(dim=(2,3), keepdim=True)         # (1, C, 1, 1)
        cam = F.relu((weights * acts).sum(dim=1, keepdim=True))  # (1,1,h,w)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_heatmap_gray_tensor(gray_tensor, cam_map,
                                out_path='cam_gray_tensor_overlay.jpg',
                                mean=0.5, std=0.5):
    """
    gray_tensor: torch.Tensor, shape (1,1,H,W) 或者 (1,H,W)
                 值域为归一化后（如 Normalize([mean],[std])）的浮点数
    cam_map:     np.ndarray, shape (H,W)，值域 [0,1]
    mean, std:   归一化时用到的均值和方差，默认和前面示例一致
    out_path:    保存路径
    """
    # 1) 把 tensor 转成 numpy 灰度图（0-255 uint8）
    t = gray_tensor.detach().cpu()
    # squeeze 到 (H,W)
    if t.dim() == 4:
        t = t.squeeze(0)  # (1,H,W)
    if t.dim() == 3:
        t = t.squeeze(0)  # (H,W)
    # 反归一化： x = x_norm * std + mean
    gray = t * std + mean
    # 转 0-255
    gray = (gray * 255.0).clamp(0, 255).numpy().astype(np.uint8)

    # 2) 若 cam_map 大小与 gray 不同，可先调整
    if cam_map.shape != gray.shape:
        cam_map_resized = cv2.resize(cam_map, (gray.shape[1], gray.shape[0]))
    else:
        cam_map_resized = cam_map

    # 3) 生成热力图并叠加
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_map_resized), cv2.COLORMAP_JET)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(gray_bgr, 0.6, heatmap, 0.4, 0)

    # 4) 保存并返回
    cv2.imwrite(out_path, overlay)
    return overlay

if __name__ == '__main__':
    pass
    # # 1) 构建单通道 ResNet50
    # model = make_resnet1ch(pretrained=True)
    # # 2) 选择可视化层（以 layer4 最后一个 Bottleneck 的 conv3 为例）
    # target_layer = model.layer4[-1].conv3
    # gradcam = GradCAM(model, target_layer)
    #
    # # 3) 读入灰度图
    # img_path = 'your_gray_image.jpg'
    # inp = preprocess_gray(img_path)  # (1,1,224,224)
    #
    # # 4) 生成 CAM
    # cam = gradcam(inp)
    #
    # # 5) 叠加并保存
    # vis = overlay_heatmap_gray(img_path, cam, out_path='cam_gray_out.jpg')
    # print("灰度图 Grad-CAM 结果保存在 cam_gray_out.jpg")