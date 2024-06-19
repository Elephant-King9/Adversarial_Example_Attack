import torch
import torch.nn.functional as F

from utils.denorm import denorm


class attack_MIFGSM:
    def __init__(self, model, config):
        self.config = config
        self.model = model

    def attack(self, image, epsilon, label):
        perturbed_image = image.clone().detach().to(self.config.device)
        # 反向归一化处理
        perturbed_image = denorm(perturbed_image, self.config.device)
        # 创建一个shape和perturbed_image相同的tensor类型张量，但是所有参数都为0
        # 相当于公式中的g~0~参数，论文要求一开始初始化为0
        momentum = torch.zeros_like(perturbed_image).to(self.config.device)
        for _ in range(epsilon):
            perturbed_image = perturbed_image.clone().detach().to(self.config.device)
            perturbed_image.requires_grad = True
            output = self.model(perturbed_image)
            loss = F.nll_loss(output, label)
            self.model.zero_grad()
            loss.backward()

            # 计算梯度
            grad = perturbed_image.grad.data
            # 相当于计算公式中的分数部分的值
            # grad.abs().sum(dim=(1, 2, 3), keepdim=True)相当于计算分母，也就是梯度的L1范式
            normalized_grad = grad / grad.abs().sum(dim=(1, 2, 3), keepdim=True)
            momentum = self.config.momentum * momentum + normalized_grad
            perturbed_image = perturbed_image + self.config.alpha * momentum.sign()
            perturbed_image = torch.clamp(perturbed_image, 0, 1).detach()

        return perturbed_image
