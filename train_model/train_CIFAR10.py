import os
import time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from models.model_CIFAR10 import model_CIFAR10
from utils.judge_device import judge_device
from utils.mkdir import mkdir
from log_config import logger


class train_CIFAR10:
    def __init__(self, config):
        self.config = config

    def train(self):
        # 创建数据集
        train_dataset = torchvision.datasets.CIFAR10('./assets/datasets', train=True, download=False,
                                                     transform=torchvision.transforms.ToTensor())
        # 创建DataLoader
        train_dataLoader = DataLoader(dataset=train_dataset, batch_size=64)
        # 创建自定义的神经网络
        model = model_CIFAR10().to(self.config.device)

        # 加载损失函数与梯度下降算法
        loss_fn = nn.CrossEntropyLoss().to(self.config.device)
        learn_rate = 1e-2
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)
        # 定义训练网络的时的变量
        # 循环轮数
        epoch = 20
        # 总训练次数
        total_train_step = 0
        # 记录开始时间
        start_time = time.time()

        # 保存路径
        save_path = os.path.join(self.config.pre_train_path, 'CIFAR10')
        mkdir(save_path)
        save_path = judge_device(self.config.device, save_path)
        logger.info(f'Save Path is:{save_path}')

        for i in range(epoch):
            # 更新模型为训练模式
            model.train()
            # 定义每轮的总训练损失
            pre_train_loss = 0.0
            # 定义每轮的训练次数
            pre_train_step = 0
            for data in train_dataLoader:
                inputs, labels = data
                # 使用GPU进行训练
                inputs = inputs.to(self.config.device)
                labels = labels.to(self.config.device)
                # 优化器清零
                optimizer.zero_grad()
                # 根据模型将输入转化为输出
                outputs = model(inputs)
                # 计算损失，并且找到最大的下降梯度
                loss = loss_fn(outputs, labels)
                loss.backward()
                # 优化器进行梯度下降更新
                optimizer.step()
                # 参数更新，用于TensorBoard与print
                pre_train_loss += loss.item()
                pre_train_step += 1
                total_train_step += 1

                # 每循环100次输出一下
                if pre_train_step % 100 == 0:
                    end_time = time.time()
                    print(
                        f'Epoch:{i + 1},Step:{pre_train_step},Loss:{pre_train_loss / pre_train_step},Time:{end_time - start_time}')

                    # 更新到TensorBoard
            torch.save(model.state_dict(), os.path.join(save_path, f'model_CIFAR10_{i + 1}.pth'))


