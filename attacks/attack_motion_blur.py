import numpy as np
import torch
from log_config import logger
import cv2
from PIL import Image as PILImage
from io import BytesIO
from scipy.ndimage.interpolation import map_coordinates
from wand.image import Image as WandImage
from wand.api import library as wandlibrary

# 运动模糊攻击
class attack_motion_blur:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):

        image = image * 255
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        logger.debug(f'image shape before:{image.shape}')  # image shape:(1, 3, 480, 480)
        image = image.squeeze(0)
        image = np.array(image).transpose((1, 2, 0))
        logger.debug(f'image shape after:{image.shape}')  # image shape after:(480, 480, 3)

        c = [(10, 3), (15, 5), (15, 8), (15, 12), (20, 15)][epsilon - 1]

        image = np.array(image, dtype=np.float32) / 255.
        image = np.random.normal(size=image.shape[:2], loc=c[0], scale=c[1])
        # print("input:", x.shape)
        image = PILImage.fromarray((np.clip(image.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        # print(x.size)

        output = BytesIO()
        image.save(output, format='PNG')
        image = MotionImage(blob=output.getvalue())

        # x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        image.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        # print("x.motion_blur:", x.size)

        image = cv2.imdecode(np.fromstring(image.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)
        perturbed_image =  np.clip(np.array([image, image, image]).transpose((1, 2, 0)), 0, 255) / 255
        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(
            f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image

class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0, **kwargs):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
