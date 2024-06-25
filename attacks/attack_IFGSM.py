import torch
import time
from torchvision import transforms

from utils.denorm import denorm


class attack_IFGSM:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def attack(self, image, epsilon, label):
        """
        Perform IFGSM attack
        :param image: 输入图片
        :param epsilon: 迭代的轮数
        :param label: 标签
        :return: 生成的对抗样本
        """
        # 反向归一化处理
        perturbed_image = denorm(image, self.config.device)

        # 进行多轮迭代，迭代的次数为当前的epsilon值
        for _ in range(epsilon):
            sign_data_grad = self.model.calc_image_grad(perturbed_image, label).sign()
            perturbed_image = torch.clamp(perturbed_image + self.config.alpha * sign_data_grad, 0, 1).detach()

        return perturbed_image
