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
        # abort_early如果为 True，则启用提前终止机制。
        self.abort_early = config.ABORT_EARLY  # 提前终止
        self.initial_const = config.INITIAL_CONST  # 初始常数c
        self.targeted = config.TARGETED  # 是否进行目标攻击

    def attack(self, image, epsilon, label, **kwargs):
        """
        Perform CW attack on classification
        :param image: 输入图片
        :param label: 标签（目标标签或原标签）
        :param epsilon: 迭代次数
        """
        image = image.clone().detach().to(self.device)
        perturbed_image = image.clone().detach().requires_grad_(True)
        # batch_size = image.size(0)
        logger.debug(f'label:{check_encoding_type(label)}')
        # 初始化变量
        best_perturbed_image = image.clone()
        best_loss = float('inf')

        # 使用二分查找修改c的值
        # c的下界
        lower_bound = torch.zeros(self.batch_size).to(self.device)
        # 初始化c
        const = torch.ones(self.batch_size).to(self.device) * self.initial_const
        # c的上界
        upper_bound = torch.ones(self.batch_size).to(self.device) * 1e10

        # 通过多次二分搜索来调整常数 const，以找到最佳的扰动权衡。
        for binary_search_step in range(self.binary_search_steps):
            optimizer = optim.Adam([perturbed_image], lr=self.learning_rate)
            prev_loss = float('inf')

            # 初始化 loss1 和 loss2
            loss1 = torch.zeros(self.batch_size).to(self.device)
            loss2 = torch.zeros(self.batch_size).to(self.device)

            for iteration in range(epsilon):
                optimizer.zero_grad()

                # 计算对抗样本的预测输出
                outputs = self.model.predict(perturbed_image)

                # logger.debug(f'Outputs: {outputs}')
                # 计算损失函数
                # label 是目标标签的one-hot编码（在目标攻击中）。
                # label * outputs 会提取 outputs 中与目标标签对应的部分。
                # real用于获取真实标签的logit
                real = torch.sum(label * outputs, dim=1)
                # other用于获得除真实标签以外的最大logit值
                other = torch.max((1 - label) * outputs - (label * 1e4), dim=1)[0]
                
                # logger.debug(f'Real: {real}, Other: {other}')

                if self.targeted:
                    # 目标攻击的目的是使模型的预测结果变为特定的目标类别，即希望模型将输入图像分类为攻击者指定的目标类别。
                    loss1 = torch.clamp(other - real + self.confidence, min=0)
                else:
                    # 非目标攻击的目的是使模型的预测结果变为除真实类别以外的任意其他类别，即希望模型的预测结果不再是原来的真实类别。
                    loss1 = torch.clamp(real - other + self.confidence, min=0)

                l2_loss = torch.sum((perturbed_image - image) ** 2, dim=[1, 2, 3])
                loss2 = l2_loss
                loss = torch.sum(const * loss1 + loss2)
                logger.debug(f'epslion:{epsilon}, loss1:{loss1.item()}, loss2:{loss2.item()}, Loss:{loss.item()}')


                # 反向传播和优化
                loss.backward()

                optimizer.step()

                # 提前终止
                # 如果损失函数在多个迭代中没有显著变化，则认为模型已经收敛，可以提前终止训练。
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
                    perturbed_image.data = torch.clamp(perturbed_image, 0, 1)

            # 更新常数
            for i in range(self.batch_size):
                if (loss1[i] == 0) and (loss2[i] < best_loss):
                    # loss1 == 0代表成功地让模型做出错误预测
                    # loss2 < best_loss又与原始样本尽可能接近,l2损失最小

                    # 重新更新c的上界
                    upper_bound[i] = min(upper_bound[i], const[i])
                    if upper_bound[i] < 1e9:
                        # 取上下界的中间值
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = max(lower_bound[i], const[i])
                    if upper_bound[i] < 1e9:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        const[i] *= 10
        return best_perturbed_image
