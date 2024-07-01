import numpy as np
import torch
from skimage.filters import gaussian
from log_config import logger


# 高斯模糊攻击
class attack_gaussian_blur:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()

        c = [1, 2, 3, 4, 6][epsilon - 1]

        image = gaussian(np.array(image), sigma=c)
        perturbed_image = np.clip(image, 0, 1)
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        logger.debug(f'perturbed_image shape{perturbed_image.shape}')
        return perturbed_image
