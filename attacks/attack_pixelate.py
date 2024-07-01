import numpy as np
import torch
from log_config import logger
import cv2


# 像素攻击
class attack_pixelate:
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

        h = image.shape[0]
        w = image.shape[1]
        c = [0.6, 0.5, 0.4, 0.3, 0.25][epsilon - 1]

        # x = x.resize((int(224 * c), int(224 * c)), PILImage.BOX)
        image = cv2.resize(image, (int(224 * c), int(224 * c)), interpolation=cv2.INTER_AREA)
        # x = x.resize((224, 224), PILImage.BOX)
        perturbed_image = cv2.resize(image, (h, w), interpolation=cv2.INTER_AREA)

        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(
            f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image
