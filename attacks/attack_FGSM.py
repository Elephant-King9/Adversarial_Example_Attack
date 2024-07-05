import time

import torch
from torchvision import transforms
from utils.denorm import denorm


class attack_FGSM:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    # 用于生成对抗样本
    def attack(self, image, epsilon, label, **kwargs):
        """
        Perform FGSM with
        :param image: 输入图片
        :param epsilon: 𝜀超参数
        :param label: 标签
        :return:
        """
        # 恢复图片到原始尺度,进行反归一化
        # perturbed_image = denorm(image, self.config.device)
        perturbed_image = image

        # 获取梯度方向
        sign_data_grad = self.model.calc_image_grad(perturbed_image, label).sign()
        # 对反归一化的图像添加扰动
        perturbed_image = perturbed_image + epsilon * sign_data_grad
        # 将生成的对抗样本的扰动控制在0~1之间
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
