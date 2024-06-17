import os.path

import torch
from log_config import logger
# 预训练模型的路径
model_MNIST_path = './assets/Pre-training_files/model_MNIST_10.pth'


def get_model(config):
    # 选择模型
    # 自定义的MNIST模型，model_MNIST.py文件中定义
    if config.model == 'MNIST':
        from models.model_MNIST import model_MNIST
        # 如果预训练模型未找到 则加载文件
        if os.path.exists(model_MNIST_path):
            logger.info('MNIST model loaded')
            # 加载字典形式的预训练模型，并使用GPU训练
            # 定义和导入参数这两句还得分开写
            model = model_MNIST().to(config.device)
            model.load_state_dict(torch.load(model_MNIST_path))
            return model
        else:
            # 预训练模型未找到
            logger.critical('model_MNIST_path model not find')
            exit()
    else:
        logger.critical('Model not recognized')
        exit()
    return
