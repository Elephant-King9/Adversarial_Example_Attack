import os
import time

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader

from models.model_MNIST import model_MNIST
from utils.judge_device import judge_device
from utils.mkdir import mkdir
from log_config import logger


class train_MNIST:
    def __init__(self, config):
        self.config = config

    def train(self):
        train_dataset = torchvision.datasets.MNIST('./assets/datasets', train=True, download=False, transform=torchvision.transforms.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        # 配置模型框架
        model = model_MNIST().to(self.config.device)
        loss_fn = nn.CrossEntropyLoss().to(self.config.device)
        learn_rate = 1e-2
        optimizer = torch.optim.SGD(model.parameters(), lr=learn_rate)

        epoch = 10
        total_train_step = 0
        start_time = time.time()
        # 保存路径
        save_path = os.path.join(self.config.pre_train_path, 'MNIST')
        mkdir(save_path)
        save_path = judge_device(self.config.device, save_path)
        logger.info(f'Save Path is:{save_path}')
        # 训练
        for i in range(epoch):
            pre_train_step = 0
            pre_train_loss = 0
            model.train()
            for data in train_loader:
                # print(data)
                inputs, labels = data
                inputs, labels = inputs.to(self.config.device), labels.to(self.config.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()
                pre_train_step += 1
                total_train_step += 1
                pre_train_loss += loss.item()
                if pre_train_step % 100 == 0:
                    end_time = time.time()
                    print(
                        f'Epoch:{i + 1},pre_train_loss:{pre_train_loss / pre_train_step},time = {end_time - start_time}')
            torch.save(model.state_dict(), os.path.join(save_path, f'model_MNIST_{i + 1}.pth'))



