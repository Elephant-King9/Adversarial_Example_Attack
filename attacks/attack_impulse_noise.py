import numpy as np
import torch
import skimage as sk


# 脉冲噪声攻击
class attack_impulse_noise:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label, **kwargs):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()

        c = [.03, .06, .09, 0.17, 0.27][epsilon - 1]

        image = sk.util.random_noise(np.array(image), mode='s&p', amount=c)
        perturbed_image = np.clip(image, 0, 1)
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)

        return perturbed_image
