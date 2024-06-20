import os.path

import torchvision

from log_config import logger

dataset_MNIST_path = './assets/datasets'


def get_dataset(config):
    # MNIST数据集
    if config.dataset == 'MNIST':
        # 判断数据集是否存在
        if os.path.isdir(os.path.join(dataset_MNIST_path, 'MNIST')):
            train_dataset = torchvision.datasets.MNIST(dataset_MNIST_path, train=True, download=config.download,
                                                       transform=config.transform)
            val_dataset = torchvision.datasets.MNIST(dataset_MNIST_path, train=False, download=config.download,
                                                     transform=config.transform)
            logger.info('MNIST dataset loaded')
            return train_dataset, val_dataset
        else:
            logger.critical('MNIST dataset not found.')
            exit()

    else:
        logger.critical('Dataset {args.dataset} not recognized')
        exit()

    return



