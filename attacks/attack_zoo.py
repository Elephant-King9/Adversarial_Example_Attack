from log_config import logger


# 获取目标模型
def get_attack(model, val_DataLoader, config):
    if config.attack == 'FGSM':
        from attacks.attack_FGSM import attack_FGSM
        attacker = attack_FGSM(model, config)
        config.epsilons = [0, .05, .1, .15, .2, .25, .3]
        logger.info('FGSM attack loaded')
        logger.info(f'FGSM epsilons: {config.epsilons}')
        return attacker
    elif config.attack == 'IFGSM':
        from attacks.attack_IFGSM import attack_IFGSM
        attacker = attack_IFGSM(model, config)
        config.epsilons = [0, 5, 10, 15, 20]
        logger.info('IFGSM attack loaded')
        logger.info(f'IFGSM iters: {config.epsilons}')
        return attacker
    elif config.attack == 'MIFGSM':
        from attacks.attack_MIFGSM import attack_MIFGSM
        attacker = attack_MIFGSM(model, config)
        config.epsilons = [0, 5, 10, 15, 20]
        logger.info('IFGSM attack loaded')
        logger.info(f'IFGSM iters: {config.epsilons}')
        return attacker
    elif config.attack == 'gaussian_noise':
        from attacks.attack_gaussian_noise import attack_gaussian_noise
        attacker = attack_gaussian_noise(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Gaussian noise attack loaded')
        logger.info(f'Gaussian noise iters: {config.epsilons}')
        return attacker
    elif config.attack == 'shot_noise':
        from attacks.attack_shot_noise import attack_shot_noise
        attacker = attack_shot_noise(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Shot noise attack loaded')
        logger.info(f'Shot noise iters: {config.epsilons}')
        return attacker
    elif config.attack == 'impulse_noise':
        from attacks.attack_impulse_noise import attack_impulse_noise
        attacker = attack_impulse_noise(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Impulse noise attack loaded')
        logger.info(f'Impulse noise iters: {config.epsilons}')
        return attacker
    else:
        logger.critical(f'Attack {config.attacker} not recognized')
        exit()
