import time

import torch
from torchvision import transforms
from utils.denorm import denorm


class attack_FGSM:
    def __init__(self, model, val_DataLoader, config):
        self.model = model
        self.device = config.device
        self.val_DataLoader = val_DataLoader

    # 用于生成对抗样本
    def attack(self, image, epsilon, data_grad):
        """
        Perform FGSM with
        :param image: 输入图片
        :param epsilon: 𝜀超参数
        :param data_grad: 梯度
        :return:
        """
        # 克隆原始图像，以免修改原图
        # 这里是将原图clone下来，且与原图的梯度分离，最大限度的保证原图不受影响
        perturbed_image = image.clone().detach().to(self.device)

        # 获取梯度方向
        sign_data_grad = data_grad.sign()
        # 对原始图像添加扰动
        perturbed_image = image + epsilon * sign_data_grad
        # 将生成的对抗样本的扰动控制在0~1之间
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
