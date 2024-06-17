from attacks.attack_FGSM import attack_FGSM
from log_config import logger


# 获取目标模型
def get_attack(model, val_DataLoader, config):
    if config.attack == 'FGSM':
        attacker = attack_FGSM(model, val_DataLoader, config)
        logger.info('FGSM attack loaded')
        return attacker
    else:
        logger.critical(f'Attack {config.attacker} not recognized')
        exit()
