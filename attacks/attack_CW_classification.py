import torch
import torch.nn.functional as F


class attack_CW_classification:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.c = config.c
        self.lr = config.lr

    def attack(self, image, epsilon, label):
        """
        Perform CW attack
        :param image: 输入图片
        :param epsilon: 迭代轮数
        :param label: 标签
        """
        # 初始化变量
        image = image.clone().detach().to(self.config.device)
        label = label.to(self.config.device)
        perturbed_image = image.clone().detach()
        perturbed_image.requires_grad = True

        optimizer = torch.optim.Adam([perturbed_image], lr=self.lr)

        for _ in range(epsilon):
            optimizer.zero_grad()

            output = self.model.predict(perturbed_image)
            # 提取模型真实标签
            real = output.gather(1, label.unsqueeze(1)).squeeze(1)
            # 提取模型对除真实标签之外的类别的最高置信度
            other = output.max(1)[0]

            # CW目标函数
            # 相当于f6 , > 0 的时候说明还没有被预测错误，
            f_loss = torch.clamp(real - other, min=0)
            # 计算L2损失
            l2_loss = F.mse_loss(perturbed_image, image)
            # 总公式，目的是最小化这个值
            loss = self.c * f_loss + l2_loss

            loss.backward()
            optimizer.step()

            # 将生成的对抗样本的扰动控制在0~1之间
            perturbed_image.data = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image
