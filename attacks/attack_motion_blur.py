import numpy as np
import torch
from matplotlib import pyplot as plt

from log_config import logger
import cv2
from PIL import Image as PILImage
from io import BytesIO
from scipy.ndimage.interpolation import map_coordinates
from wand.image import Image as WandImage
from wand.api import library as wandlibrary


# 运动模糊攻击
class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)


# 运动模糊攻击
class attack_motion_blur:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):
        # 恢复到 [0, 255] 范围
        image = image * 255
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        image = image.squeeze(0)
        image = np.array(image).transpose((1, 2, 0))

        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][epsilon - 1]

        x = np.array(image, dtype=np.float32) / 255.
        x = np.clip(x, 0, 1)
        x = PILImage.fromarray((x * 255).astype(np.uint8))

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        x = cv2.imdecode(np.frombuffer(x.make_blob(), np.uint8), cv2.IMREAD_UNCHANGED)

        # 将灰度图像转换为RGB
        if len(x.shape) == 2:  # 如果是单通道灰度图像
            x = np.stack([x] * 3, axis=-1)

        perturbed_image = np.clip(x, 0, 255)
        perturbed_image = perturbed_image / 255.0  # 归一化到 [0, 1]
        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        print(f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image
