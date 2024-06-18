from log_config import logger


# 根据不同的攻击方式绘制图片
def get_picture(config):
    if config.dataset == 'MNIST' and config.attack == 'FGSM':
        from pictures.picture_MNIST_FSGM import Picture_MNIST_FSGM
        logger.info('print pictures')
        picture = Picture_MNIST_FSGM(config)
        picture.draw()
    elif config.dataset == 'MNIST' and config.attack == 'IFGSM':
        from pictures.picture_MNIST_IFGSM import Picture_MNIST_IFGSM
        logger.info('print pictures')
        picture = Picture_MNIST_IFGSM(config)
        picture.draw()
    else:
        logger.warning('don\'t have pictures')
