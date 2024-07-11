import torch
import torch.optim as optim
import numpy as np

class attack_CW:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.confidence = config.CONFIDENCE         # 置信度
        self.learning_rate = config.LEARNING_RATE   # 学习率
        self.max_iterations = config.MAX_ITERATIONS # 最大迭代次数
        self.binary_search_steps = config.BINARY_SEARCH_STEPS   # 二分搜索步数
        self.abort_early = config.ABORT_EARLY       # 早起终止
        self.initial_const = config.INITIAL_CONST   # 初始常数c
        self.boxmin = config.BOXMIN                 # 像素的最小值
        self.boxmax = config.BOXMAX                 # 像素的最大值
        self.targeted = config.TARGETED             # 是否进行目标攻击

    def attack(self, image, label, epsilon, **kwargs):
        """
        Perform CW attack on classification
        :param image: 输入图片
        :param label: 标签（目标标签或原标签）
        :param epsilon: 迭代次数
        """
        image = image.clone().detach().to(self.device)
        perturbed_image = image.clone().detach().requires_grad_(True)
        batch_size = image.size(0)

        # 初始化变量
        best_perturbed_image = image.clone()
        best_loss = float('inf')

        lower_bound = torch.zeros(batch_size).to(self.device)
        const = torch.ones(batch_size).to(self.device) * self.initial_const
        upper_bound = torch.ones(batch_size).to(self.device) * 1e10

        for binary_search_step in range(self.binary_search_steps):
            optimizer = optim.Adam([perturbed_image], lr=self.learning_rate)
            prev_loss = float('inf')

            for iteration in range(epsilon):
                optimizer.zero_grad()

                # 计算对抗样本的预测输出
                outputs = self.model(perturbed_image)

                # 计算损失函数
                real = torch.sum(label * outputs, dim=1)
                other = torch.max((1 - label) * outputs - (label * 1e4), dim=1)[0]

                if self.targeted:
                    loss1 = torch.clamp(other - real + self.confidence, min=0)
                else:
                    loss1 = torch.clamp(real - other + self.confidence, min=0)

                l2_loss = torch.sum((perturbed_image - image) ** 2, dim=[1, 2, 3])
                loss2 = l2_loss
                loss = torch.sum(const * loss1 + loss2)

                # 反向传播和优化
                loss.backward()
                optimizer.step()

                # 提前终止
                if self.abort_early and iteration % (epsilon // 10) == 0:
                    if loss > prev_loss * 0.9999:
                        break
                    prev_loss = loss

                # 更新最佳结果
                if loss < best_loss:
                    best_loss = loss
                    best_perturbed_image = perturbed_image.clone().detach()

                # 将对抗样本的像素值限制在合法范围内
                with torch.no_grad():
                    perturbed_image.data = torch.clamp(perturbed_image, self.boxmin, self.boxmax)

            # 更新常数
            for i in range(batch_size):
                if (loss1[i] == 0) and (loss2[i] < best_loss):
                    upper_bound[i] = min(upper_bound[i], const[i])
                    if upper_bound[i] < 1e9:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = max(lower_bound[i], const[i])
                    if upper_bound[i] < 1e9:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        const[i] *= 10

        return best_perturbed_image

class Config:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIDENCE = 0
    LEARNING_RATE = 1e-2
    MAX_ITERATIONS = 10000
    BINARY_SEARCH_STEPS = 9
    ABORT_EARLY = True
    INITIAL_CONST = 1e-3
    BOXMIN = 0.0  # assuming pixel values range from 0 to 1
    BOXMAX = 1.0
    TARGETED = True

# 示例用法
# model = YourModel()
# config = Config()
# attacker = attack_CW(model, config)
# image = torch.randn(1, 3, 224, 224)  # 示例输入图像
# label = torch.tensor([1, 0, 0])  # 示例目标标签
# epsilon = 1000  # 示例迭代次数
# perturbed_image = attacker.attack(image, label, epsilon)
