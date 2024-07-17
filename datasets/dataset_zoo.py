import os.path

import torchvision

from log_config import logger
from datasets.dataset_blip_caption import DatasetCaption

dataset_path = './assets/datasets'


def get_dataset(config):
    # MNIST数据集
    if config.dataset == 'MNIST':
        # 判断数据集是否存在
        if os.path.isdir(os.path.join(dataset_path, 'MNIST')):
            # train_dataset = torchvision.datasets.MNIST(dataset_MNIST_path, train=True, download=config.download,
            #                                            transform=config.transform)
            val_dataset = torchvision.datasets.MNIST(dataset_path, train=False, download=config.download,
                                                     transform=config.transform)
            logger.info('MNIST dataset loaded')
            return val_dataset
        else:
            logger.critical('MNIST dataset not found.')
            exit()
    elif config.dataset == 'coco':
        data_dir = '././assets/datasets'
        val_dataset = DatasetCaption(config, data_dir=data_dir)
        return val_dataset
    elif config.dataset == 'CIFAR10':
        # 判断数据集是否存在
        if os.path.isdir(os.path.join(dataset_path, 'cifar-10')):
            # train_dataset = torchvision.datasets.MNIST(dataset_MNIST_path, train=True, download=config.download,
            #                                            transform=config.transform)
            val_dataset = torchvision.datasets.CIFAR10(dataset_path, train=False, download=config.download,
                                                       transform=config.transform)
            logger.info('CIFAR10 dataset loaded')
            return val_dataset
        else:
            logger.critical('CIFAR10 dataset not found.')
            exit()

    else:
        logger.critical('Dataset {args.dataset} not recognized')
        exit()

    return
