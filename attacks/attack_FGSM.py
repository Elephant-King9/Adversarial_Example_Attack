import time

import torch
from torchvision import transforms
from utils.denorm import denorm


class attack_FGSM:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    # 用于生成对抗样本
    def attack(self, image, epsilon, label):
        """
        Perform FGSM with
        :param image: 输入图片
        :param epsilon: 𝜀超参数
        :param label: 标签
        :return:
        """

        # 计算梯度，反向传播
        # 这里并没有对原图进行更新，仅仅是计算了原图的梯度
        # 这里去问了问GPT说是反归一化前后的梯度计算是不同的，但是用归一化之前的梯度去添加扰动的效果更好
        # 这里为了少传递一个参数output重新计算了一下梯度，可能会影响速度，但是可以让传入的参数变少
        loss = torch.nn.functional.nll_loss(self.model(image), label)
        self.model.zero_grad()
        loss.backward()

        # 收集图片梯度
        data_grad = image.grad.data
        # 恢复图片到原始尺度,进行反归一化
        data_denorm = denorm(image, self.config.device)

        # 获取梯度方向
        sign_data_grad = data_grad.sign()
        # 对反归一化的图像添加扰动
        perturbed_image = data_denorm + epsilon * sign_data_grad
        # 将生成的对抗样本的扰动控制在0~1之间
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
