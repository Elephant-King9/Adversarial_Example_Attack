import os.path

import torch
from log_config import logger
from utils.judge_device import judge_device


# 预训练模型的路径


def get_model(config):
    # 选择模型
    # 自定义的MNIST模型，model_MNIST.py文件中定义
    if config.model == 'MNIST':
        from models.model_MNIST import model_MNIST
        # 如果预训练模型未找到 则加载文件
        save_path = os.path.join(config.pre_train_path, 'MNIST')
        save_path = judge_device(config.device, save_path)
        # 最终训练模型的文件
        save_path = os.path.join(save_path, 'model_MNIST_10.pth')

        if os.path.exists(save_path):
            logger.info('MNIST model loaded')
        else:
            from train_model.train_MNIST import train_MNIST
            # 预训练模型未找到
            logger.info('model_MNIST_path model not find,pretrain start')
            train_MNIST = train_MNIST(config)
            train_MNIST.train()

        # 加载字典形式的预训练模型，并使用GPU训练
        # 定义和导入参数这两句还得分开写
        model = model_MNIST().to(config.device)
        model.load_state_dict(torch.load(save_path))
        return model

    else:
        logger.critical('Model not recognized')
        exit()
    return
