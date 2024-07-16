import torch
import torch.optim as optim
import numpy as np
from log_config import logger
from utils.check_encoding_type import *


class attack_CW_classification:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.batch_size = config.batch_size
        self.confidence = config.CONFIDENCE  # 置信度
        self.learning_rate = config.LEARNING_RATE  # 学习率
        self.binary_search_steps = config.BINARY_SEARCH_STEPS  # 二分搜索步数
        self.abort_early = config.ABORT_EARLY  # 提前终止
        self.initial_const = config.INITIAL_CONST  # 初始常数c
        self.targeted = config.TARGETED  # 是否进行目标攻击
        self.num_classes = 10  # 标签分类数，先用MNIST，默认为10

    def attack(self, image, epsilon, label, **kwargs):
        """
        Perform CW attack on classification
        :param image: 输入图片
        :param label: 标签（目标标签或原标签）
        :param epsilon: 迭代次数
        """
        # logger.info('----------------------------begin----------------------------')
        image = image.clone().detach().to(self.device)
        perturbed_image = image.clone().detach().requires_grad_(True)
        label = self.to_one_hot(label, self.num_classes)

        # 初始化变量
        best_perturbed_image = image.clone()
        best_loss = float('inf')

        lower_bound = torch.zeros(self.batch_size).to(self.device)
        const = torch.ones(self.batch_size).to(self.device) * self.initial_const
        upper_bound = torch.ones(self.batch_size).to(self.device) * 1e10

        for binary_search_step in range(self.binary_search_steps):
            logger.debug(f'const:{const.item()}')
            optimizer = optim.Adam([perturbed_image], lr=self.learning_rate)
            prev_loss = float('inf')

            loss1 = torch.zeros(self.batch_size).to(self.device)
            loss2 = torch.zeros(self.batch_size).to(self.device)

            for iteration in range(epsilon):
                optimizer.zero_grad()

                outputs = self.model.predict(perturbed_image)

                real = torch.sum(label * outputs, dim=1)
                other = torch.max((1 - label) * outputs - (label * 1e4), dim=1)[0]

                # logger.debug(f'Real: {real.item()}, Other: {other.item()}')

                if self.targeted:
                    loss1_raw = other - real + self.confidence
                    loss1 = torch.clamp(loss1_raw, min=0)
                    # logger.debug(f'loss1_raw (targeted): {loss1_raw}, loss1 (clamped): {loss1}')
                else:
                    loss1_raw = real - other + self.confidence
                    loss1 = torch.clamp(loss1_raw, min=0)
                    # logger.debug(f'loss1_raw (untargeted): {loss1_raw}, loss1 (clamped): {loss1}')

                l2_loss = torch.sum((perturbed_image - image) ** 2, dim=[1, 2, 3])
                loss2 = l2_loss
                logger.debug(f'l2_loss: {l2_loss.item()}')
                loss = torch.sum(const * loss1 + loss2)
                logger.debug(f'epslion:{epsilon}, loss1:{loss1.item()}, loss2:{loss2.item()}, Loss:{loss.item()}')

                loss.backward()
                optimizer.step()

                if self.abort_early and iteration % (epsilon // 10) == 0:
                    if loss > prev_loss * 0.9999:
                        break
                    prev_loss = loss

                if loss < best_loss and loss1 == 0:
                    best_loss = loss
                    best_perturbed_image = perturbed_image.clone().detach()
                    logger.debug(f'Best loss:{best_loss}')

                with torch.no_grad():
                    perturbed_image.data = torch.clamp(perturbed_image, 0, 1)

            for i in range(self.batch_size):
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

    def to_one_hot(self, y, num_classes):
        # 先将y移动到CPU，然后生成one-hot编码，最后移动到原始设备上
        y_cpu = y.to('cpu')
        one_hot = torch.eye(num_classes)[y_cpu].to(y.device)
        return one_hot