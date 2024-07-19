# 主函数
import time

import torch
from torch.utils.data import DataLoader

from task_classification import task_classification
from task_caption import task_caption
from attacks.attack_zoo import get_attack
from datasets.dataset_zoo import get_dataset
from models.model_zoo import get_model
from log_config import logger


def main(config):
    sum_start_time = time.time()
    logger.info(f"train device is {config.device}")
    # 根据config中的model字段获取模型
    model = get_model(config)
    # 根据config中的dataset字段获取测试集
    val_dataset = get_dataset(config)
    # 获取dataLoader
    val_dataLoader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    # 根据config中的attack字段获取攻击模型
    attacker = get_attack(model, val_dataLoader, config)

    logger.info("Start attacking...")
    # 用于记录训练轮数
    i = 0
    for eps in config.epsilons:
        if config.dataset == "CIFAR10" or config.dataset == "MNIST":
            # 代表分类任务
            data = task_classification(eps, attacker, model, val_dataLoader, config)
        elif config.dataset == "coco":
            # 代表Image Caption任务
            data = task_caption(eps, attacker, model, val_dataLoader, config)

        if len(data) == 2:
            # 分类任务
            acc, ex = data
            config.accuracies.append(acc)
        else:
            # caption任务
            ex = data
        config.examples.append(ex)
        logger.info(f"Finish {i + 1} around attacking,{len(config.epsilons) - i - 1} rounds left.")
        i = i + 1
    sum_end_time = time.time()
    logger.info(f"sum time is {sum_end_time - sum_start_time}")
