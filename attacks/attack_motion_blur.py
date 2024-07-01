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

        x = np.array(image, dtype=np.float32) / 255.
        x = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])
        # print("input:", x.shape)
        x = PILImage.fromarray((np.clip(x.squeeze(), 0, 1) * 255).astype(np.uint8), mode='L')
        # print(x.size)

        output = BytesIO()
        x.save(output, format='PNG')
        x = MotionImage(blob=output.getvalue())

        # x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        x.motion_blur(radius=c[0], sigma=c[1], angle=np.random.uniform(-45, 45))
        # print("x.motion_blur:", x.size)

        x = cv2.imdecode(np.fromstring(x.make_blob(), np.uint8),
                         cv2.IMREAD_UNCHANGED)
        # print("cv2.imdecode:", x.shape)

        # tmp = np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
        # print("tmp:", tmp.shape)

        # if x.shape != (224, 224):
        #    return np.clip(x[..., [2, 1, 0]], 0, 255)  # BGR to RGB
        # else:  # greyscale to RGB
        #    return np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
        perturbed_image =  np.clip(np.array([x, x, x]).transpose((1, 2, 0)), 0, 255)
        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(
            f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image / 255


class MotionImage(WandImage):
    def motion_blur(self, radius=0.0, sigma=0.0, angle=0.0, **kwargs):
        wandlibrary.MagickMotionBlurImage(self.wand, radius, sigma, angle)
