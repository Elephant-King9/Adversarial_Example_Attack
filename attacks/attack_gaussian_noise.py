import numpy as np
import torch


# 高斯噪声攻击
class attack_gaussian_noise:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):
        # 将tensor数据类型的图片转化为numpy
        image = image.cpu().numpy()
        # 根据eps添加不同等级的高斯噪声，c为标准差
        c = [.08, .12, 0.18, 0.26, 0.38][epsilon - 1]
        # 归一化
        image = np.array(image) / 255.
        # 将返回结果限制在0~1中
        perturbed_image = np.clip(image + np.random.normal(size=image.shape, scale=c), 0, 1)
        perturbed_image = torch.from_numpy(perturbed_image).float().to(self.config.device)
        return perturbed_image
