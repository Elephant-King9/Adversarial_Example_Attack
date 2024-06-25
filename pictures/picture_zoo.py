from log_config import logger


# 根据不同的攻击方式绘制图片
def get_picture(config):
    if config.dataset == 'MNIST':
        if config.attack == 'FGSM':
            from pictures.picture_MNIST_FGSM import Picture_MNIST_FSGM
            logger.info('print pictures')
            picture = Picture_MNIST_FSGM(config)
            picture.draw()
        elif config.attack == 'IFGSM':
            from pictures.picture_MNIST_IFGSM import Picture_MNIST_IFGSM
            logger.info('print pictures')
            picture = Picture_MNIST_IFGSM(config)
            picture.draw()
        elif config.attack == 'MIFGSM':
            from pictures.picture_MNIST_MIFGSM import Picture_MNIST_MIFGSM
            logger.info('print pictures')
            picture = Picture_MNIST_MIFGSM(config)
            picture.draw()
        else:
            logger.critical("Dont\'t find attacker to draw picture!")
            exit()
    elif config.dataset == 'coco':
        if config.attack == 'FGSM':
            from pictures.picture_COCO_FGSM import Picture_COCO_FSGM
            logger.info('print pictures')
            picture = Picture_COCO_FSGM(config)
            picture.draw()
        elif config.attack == 'IFGSM':
            from pictures.picture_COCO_IFGSM import Picture_COCO_IFGSM
            logger.info('print pictures')
            picture = Picture_COCO_IFGSM(config)
            picture.draw()
        elif config.attack == 'MIFGSM':
            from pictures.picture_COCO_MIFGSM import Picture_COCO_MIFGSM
            logger.info('print pictures')
            picture = Picture_COCO_MIFGSM(config)
            picture.draw()
        else:
            logger.critical("Dont\'t find attacker to draw picture!")
            exit()

    else:
        logger.warning('don\'t have pictures')
