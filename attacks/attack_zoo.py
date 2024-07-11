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
    elif config.attack == 'speckle_noise':
        from attacks.attack_speckle_noise import attack_speckle_noise
        attacker = attack_speckle_noise(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Speckle noise attack loaded')
        logger.info(f'Speckle noise iters: {config.epsilons}')
        return attacker
    elif config.attack == 'gaussian_blur':
        from attacks.attack_gaussian_blur import attack_gaussian_blur
        attacker = attack_gaussian_blur(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Gaussian blur attack loaded')
        logger.info(f'Gaussian blur iters: {config.epsilons}')
        return attacker
    elif config.attack == 'defocus_blur':
        from attacks.attack_defocus_blur import attack_defocus_blur
        attacker = attack_defocus_blur(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Defocus blur attack loaded')
        logger.info(f'Defocus blur iters: {config.epsilons}')
        return attacker
    elif config.attack == 'zoom_blur':
        from attacks.attack_zoom_blur import attack_zoom_blur
        attacker = attack_zoom_blur(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Zoom blur attack loaded')
        logger.info(f'Zoom blur iters: {config.epsilons}')
        return attacker
    elif config.attack == 'fog':
        from attacks.attack_fog import attack_fog
        attacker = attack_fog(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Fog attack loaded')
        logger.info(f'Fog iters: {config.epsilons}')
        return attacker
    elif config.attack == 'frost':
        from attacks.attack_frost import attack_frost
        attacker = attack_frost(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Frost attack loaded')
        logger.info(f'Frost iters: {config.epsilons}')
        return attacker
    elif config.attack == 'snow':
        from attacks.attack_snow import attack_snow
        attacker = attack_snow(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Snow attack loaded')
        logger.info(f'Snow iters: {config.epsilons}')
        return attacker
    elif config.attack == 'spatter':
        from attacks.attack_spatter import attack_spatter
        attacker = attack_spatter(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Spatter attack loaded')
        logger.info(f'Spatter iters: {config.epsilons}')
        return attacker
    elif config.attack == 'contrast':
        from attacks.attack_contrast import attack_contrast
        attacker = attack_contrast(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Contrast attack loaded')
        logger.info(f'Contrast iters: {config.epsilons}')
        return attacker
    elif config.attack == 'brightness':
        from attacks.attack_brightness import attack_brightness
        attacker = attack_brightness(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Brightness attack loaded')
        logger.info(f'Brightness iters: {config.epsilons}')
        return attacker
    elif config.attack == 'saturate':
        from attacks.attack_saturate import attack_saturate
        attacker = attack_saturate(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Saturate attack loaded')
        logger.info(f'Saturate iters: {config.epsilons}')
        return attacker
    elif config.attack == 'pixelate':
        from attacks.attack_pixelate import attack_pixelate
        attacker = attack_pixelate(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Pixelate attack loaded')
        logger.info(f'Pixelate iters: {config.epsilons}')
        return attacker
    elif config.attack == 'elastic':
        from attacks.attack_elastic import attack_elastic
        attacker = attack_elastic(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Elastic attack loaded')
        logger.info(f'Elastic iters: {config.epsilons}')
        return attacker
    elif config.attack == 'glass_blur':
        from attacks.attack_glass_blur import attack_glass_blur
        attacker = attack_glass_blur(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Glass blur attack loaded')
        logger.info(f'Glass blur iters: {config.epsilons}')
        return attacker
    elif config.attack == 'motion_blur':
        from attacks.attack_motion_blur import attack_motion_blur
        attacker = attack_motion_blur(model, config)
        config.epsilons = [1, 2, 3, 4, 5]
        logger.info('Motion blur attack loaded')
        logger.info(f'Motion blur iters: {config.epsilons}')
        return attacker
    elif config.attack == 'PGD':
        from attacks.attack_PGD import attack_PGD
        attacker = attack_PGD(model, config)
        config.epsilons = [0, 5, 10, 15, 20]
        logger.info('PGD attack loaded')
        logger.info(f'PGD iters: {config.epsilons}')
        return attacker
    elif config.attack == 'CW_classification' and config.model == 'MNIST':
        from attacks.attack_CW_classification import attack_CW_classification
        attacker = attack_CW_classification(model, config)
        config.epsilons = [0, 10, 500]
        logger.info('CW classification attack loaded')
        logger.info(f'CW iters: {config.epsilons}')
        return attacker
    elif config.attack == 'CW_caption' and config.model == 'blip_caption':
        from attacks.attack_CW_caption import attack_CW_caption
        attacker = attack_CW_caption(model, config)
        config.epsilons = [0, 5, 10, 15, 20]
        logger.info('CW Caption attack loaded')
        logger.info(f'CW iters: {config.epsilons}')
        return attacker
    elif config.attack == 'ALA_classification' and config.model == 'MNIST':
        from attacks.attack_ALA_classification import attack_ALA_classification
        attacker = attack_ALA_classification(model, config)
        config.epsilons = [0, 5, 10, 15, 20]
        logger.info('ALA classification attack loaded')
        logger.info(f'ALA iters: {config.epsilons}')
        return attacker

    else:
        logger.critical(f'Attack {config.attack} not recognized')
        exit()
