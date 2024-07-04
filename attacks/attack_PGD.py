import torch
import time
from torchvision import transforms
import numpy as np

from utils.denorm import denorm


class attack_PGD:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        # 代表扰动范围
        self.eps = config.eps

    def attack(self, image, epsilon, label):
        """
        Perform PGD attack
        :param image: 输入图片
        :param epsilon: 迭代轮数
        :param label: 标签
        :return: 生成的对抗样本
        """
        # 反向归一化处理
        # 测试，先注释掉
        # perturbed_image = denorm(image, self.config.device)
        perturbed_image = image.clone()

        # PGD 的随机初始化步骤 - 与 IFGSM 的区别 1
        random_start = torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
        perturbed_image = torch.clamp(perturbed_image + random_start, 0, 1)

        # 进行多轮迭代，迭代的次数为预设的步数
        for _ in range(epsilon):
            sign_data_grad = self.model.calc_image_grad(perturbed_image, label).sign()
            perturbed_image = perturbed_image + self.config.alpha * sign_data_grad
            # 投影步骤，确保扰动在 epsilon 限制范围内 - 与 IFGSM 的区别 3
            perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
            perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

        return perturbed_image
