from attacks.attack_FGSM import attack_FGSM
from attacks.attack_IFGSM import attack_IFGSM
from log_config import logger


# 获取目标模型
def get_attack(model, val_DataLoader, config):
    if config.attack == 'FGSM':
        attacker = attack_FGSM(model, val_DataLoader, config)
        logger.info('FGSM attack loaded')
        return attacker
    elif config.attack == 'IFGSM':
        attacker = attack_IFGSM(model, val_DataLoader, config)
        logger.info('IFGSM attack loaded')
        return attacker
    else:
        logger.critical(f'Attack {config.attacker} not recognized')
        exit()
