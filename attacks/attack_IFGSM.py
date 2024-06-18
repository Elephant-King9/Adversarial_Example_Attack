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
        # 克隆原始图像，以免修改原图
        perturbed_image = image.clone().detach().to(self.config.device)
        # 反向归一化处理
        perturbed_image = denorm(perturbed_image,self.config.device)

        # 进行多轮迭代，迭代的次数为当前的epsilon值
        for _ in range(epsilon):
            # 每次迭代克隆并设置 requires_grad
            perturbed_image = perturbed_image.clone().detach().to(self.config.device)
            perturbed_image.requires_grad = True
            # 计算损失
            loss = torch.nn.functional.nll_loss(self.model(perturbed_image), label)
            self.model.zero_grad()
            loss.backward()
            data_grad = perturbed_image.grad.data
            sign_data_grad = data_grad.sign()
            perturbed_image = torch.clamp(perturbed_image + self.config.alpha * sign_data_grad, 0, 1).detach()

        return perturbed_image
