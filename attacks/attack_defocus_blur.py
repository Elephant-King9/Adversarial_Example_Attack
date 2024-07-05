import cv2
import numpy as np
import torch
from skimage.filters import gaussian
from log_config import logger

# 散焦模糊攻击
class attack_defocus_blur:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label, **kwargs):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        logger.debug(f'image shape before:{image.shape}')   # image shape:(1, 3, 480, 480)
        image = image.squeeze(0)
        image = np.array(image).transpose((1, 2, 0))
        logger.debug(f'image shape after:{image.shape}')    # image shape after:(480, 480, 3)

        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][epsilon - 1]
        kernel = self.disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(image[:, :, d], -1, kernel))
        logger.debug(f'channels shape:{np.array(channels).shape}')  # channels shape:(3, 480, 480)
        perturbed_image = np.clip(channels, 0, 1)
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        perturbed_image = perturbed_image.unsqueeze(0)
        logger.debug(f'perturbed_image shape:{perturbed_image.shape}')  # perturbed_image shape:torch.Size([1, 3, 480, 480])
        return perturbed_image

    def disk(self, radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)
