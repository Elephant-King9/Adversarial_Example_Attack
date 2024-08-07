import os.path

import torch
from torch import nn

from log_config import logger
from utils.judge_device import judge_device



# 预训练模型的路径


def get_model(config):
    # 选择模型
    # 自定义的MNIST模型，model_MNIST.py文件中定义
    if config.model == 'MNIST':
        from models.model_MNIST import model_MNIST
        # 如果预训练模型未找到 则加载文件
        pretrained_model_path = os.path.join(config.pre_train_path, 'MNIST')
        pretrained_model_path = judge_device(config.device, pretrained_model_path)
        # 最终训练模型的文件
        pretrained_model_path = os.path.join(pretrained_model_path, 'model_MNIST_10.pth')

        if not os.path.exists(pretrained_model_path):
            from train_model.train_MNIST import train_MNIST
            # 预训练模型未找到
            logger.info('model_MNIST_path model not find,pretrain start')
            train_MNIST = train_MNIST(config)
            train_MNIST.train()

        logger.info('MNIST model loaded')

        model = model_MNIST(config, pretrained_model_path)
        return model
    # 如果是blip模型，model_blip_caption模型的定义在这里
    elif config.model == 'blip_caption':
        from models.model_blip_caption import model_blip_caption
        # 这一步包含了加载预训练模型，与.to(device)
        model = model_blip_caption(config)
        logger.info('blip_caption model loaded')
        return model
    elif config.model == 'ResNet50' and config.dataset == 'CIFAR10':
        # 暂时不可用
        from models.model_ResNet50 import model_ResNet50
        model = model_ResNet50(config)
        logger.info('ResNet50 model loaded')
        return model
    elif config.model == 'CIFAR10':
        from models.model_CIFAR10 import model_CIFAR10
        # 如果预训练模型未找到 则加载文件
        pretrained_model_path = os.path.join(config.pre_train_path, 'CIFAR10')
        pretrained_model_path = judge_device(config.device, pretrained_model_path)
        # 最终训练模型的文件
        pretrained_model_path = os.path.join(pretrained_model_path, 'model_CIFAR10_20.pth')

        if not os.path.exists(pretrained_model_path):
            from train_model.train_CIFAR10 import train_CIFAR10
            # 预训练模型未找到
            logger.info('model_CIFAR10_path model not find,pretrain start')
            train_MNIST = train_CIFAR10(config)
            train_MNIST.train()

        logger.info('CIFAR10 model loaded')

        model = model_CIFAR10(config, pretrained_model_path)
        return model
    else:
        logger.critical('Model not recognized')
        exit()
