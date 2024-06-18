# 主函数
from torch.utils.data import DataLoader

from attack_flow import attack_flow
from attacks.attack_zoo import get_attack
from datasets.dataset_zoo import get_dataset
from models.model_zoo import get_model
from pictures.picture_zoo import get_picture
from log_config import logger


def main(config):
    logger.info(f"train device is {config.device}")
    # 根据config中的model字段获取模型
    model = get_model(config)
    # 根据config中的dataset字段获取训练集、测试集
    train_dataset, val_dataset = get_dataset(config)
    # 获取dataLoader
    val_dataLoader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=config.shuffle)
    # 根据config中的attack字段获取攻击模型
    attacker = get_attack(model, val_dataLoader, config)

    # 模型切换为验证集
    model.eval()
    logger.info("Start attacking...")
    # 用于记录训练轮数
    i = 0
    for eps in config.epsilons:
        # 测试不同的扰动对于准确性的判断
        acc, ex = attack_flow(eps, attacker, model, val_dataLoader, config)
        # 将此扰动的准确度记录
        config.accuracies.append(acc)
        # 二维数组，行代表不同的epsilon，列代表当前epsilon生成的对抗样本
        config.examples.append(ex)
        logger.info(f"Finish {i + 1} around attacking,{len(config.epsilons) - i - 1} rounds left.")
        i = i + 1
    # 绘图
    get_picture(config)
