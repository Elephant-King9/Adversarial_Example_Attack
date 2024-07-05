import numpy as np
import torch


# 散斑噪声攻击
class attack_speckle_noise:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label, **kwargs):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()

        c = [.15, .2, 0.35, 0.45, 0.6][epsilon - 1]

        image = np.array(image)
        perturbed_image = np.clip(image + image * np.random.normal(size=image.shape, scale=c), 0, 1)
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        return perturbed_image
