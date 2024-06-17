from log_config import logger
from pictures.picture_MNIST_FSGM import Picture_MNIST_FSGM


# 根据不同的攻击方式绘制图片
def get_picture(config):
    if config.dataset == 'MNIST' and config.attack == 'FGSM':
        logger.info('print pictures')
        picture = Picture_MNIST_FSGM(config)
        picture.draw()
    else:
        logger.warning('don\'t have pictures')
