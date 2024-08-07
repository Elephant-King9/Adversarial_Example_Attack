# 创建自定义的神经网络
import torch
from torch import nn

from models import model_base
import torchvision.models
from torchvision.models import ResNet50_Weights


class model_ResNet50:
    def __init__(self, config=None, pretrained_model_path=''):
        self.config = config
        self.model = torchvision.models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.add_module('add_linear', nn.Linear(1000, 10))
        self.model = self.model.to(self.config.device)
        self.model.eval()

    # 计算损失
    def clac_loss(self, image, label):
        loss = torch.nn.functional.nll_loss(self.model(image), label)
        return loss

    # 计算图片梯度
    def calc_image_grad(self, image, label):
        # 将梯度与原图剥离
        image = image.clone().detach().to(self.config.device)
        # 允许获取梯度
        image.requires_grad = True
        # 计算损失
        loss = self.clac_loss(image, label)
        # 梯度清零
        self.model.zero_grad()
        # 反向传播梯度
        loss.backward()
        # 获取梯度信息
        data_grad = image.grad.data
        # 返回梯度信息
        return data_grad

    def predict(self, image):
        return self.model(image)


if __name__ == '__main__':
    model = model_ResNet50()
    print(model.model)
