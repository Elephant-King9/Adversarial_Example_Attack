import numpy as np
import torch
from log_config import logger


# 对比度攻击
class attack_contrast:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label, **kwargs):

        image = image * 255
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        logger.debug(f'image shape before:{image.shape}')  # image shape:(1, 3, 480, 480)
        image = image.squeeze(0)
        image = np.array(image).transpose((1, 2, 0))
        logger.debug(f'image shape after:{image.shape}')  # image shape after:(480, 480, 3)

        c = [0.4, .3, .2, .1, .05][epsilon - 1]

        image = np.array(image) / 255.
        means = np.mean(image, axis=(0, 1), keepdims=True)
        perturbed_image = np.clip((image - means) * c + means, 0, 1)

        perturbed_image = perturbed_image.transpose((2, 0, 1))
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(
            f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image
