import numpy as np
import torch


# 高斯噪声攻击
class attack_shot_noise:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        c = [60, 25, 12, 5, 3][epsilon - 1]
        image = np.array(image)
        perturbed_image = np.clip(np.random.poisson(image * c) / c, 0, 1)
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        return perturbed_image
