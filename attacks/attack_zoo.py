from attacks.attack_FGSM import attack_FGSM
from attacks.attack_IFGSM import attack_IFGSM
from log_config import logger


# 获取目标模型
def get_attack(model, val_DataLoader, config):
    if config.attack == 'FGSM':
        attacker = attack_FGSM(model, config)
        config.epsilons = [0, .05, .1, .15, .2, .25, .3]
        logger.info('FGSM attack loaded')
        logger.info(f'FGSM epsilons: {config.epsilons}')
        return attacker
    elif config.attack == 'IFGSM':
        attacker = attack_IFGSM(model, config)
        config.epsilons = [0, 5, 10, 15, 20]
        logger.info('IFGSM attack loaded')
        logger.info(f'IFGSM iters: {config.epsilons}')
        return attacker
    else:
        logger.critical(f'Attack {config.attacker} not recognized')
        exit()