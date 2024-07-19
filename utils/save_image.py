import os

import numpy as np
from PIL import Image

from utils.mkdir import mkdir
from log_config import logger
from matplotlib import pyplot as plt


# 保存对抗样本生成的图片
def save_image(config, adv_examples, eps):
    for i, (init_pred, final_pred, orig_ex, adv_ex) in enumerate(adv_examples):
        # 将图像归一化到0-255范围并转换为uint8类型
        adv_ex = (adv_ex * 255).astype(np.uint8)

        # 调试信息，判断为什么输出的图片是黑的
        # logger.debug(f"Image {i} min value: {adv_ex.min()}, max value: {adv_ex.max()}")

        logger.debug(f"adv_ex.shape:{adv_ex.shape}")        # MNIST : adv_ex.shape:(28, 28)

        if adv_ex.shape[0] == 3:  # 三通道图像
            adv_ex = np.transpose(adv_ex, (1, 2, 0))

        # 确保图像数据在 [0, 1] 范围内（如果是浮点数类型）
        if adv_ex.dtype == np.float32 or adv_ex.dtype == np.float64:
            adv_ex = np.clip(adv_ex, 0, 1)
        # 确保图像数据在 [0, 255] 范围内（如果是整数类型）
        elif adv_ex.dtype == np.uint8 or adv_ex.dtype == np.int32 or adv_ex.dtype == np.int64:
            adv_ex = np.clip(adv_ex, 0, 255)
        # plt.imshow(adv_ex)
        # plt.show()
        img = Image.fromarray(adv_ex)  # 使用PIL库将NumPy数组转换为图片

        # 保存对抗样本图片
        # 创建多级文件夹，防止生成结果太乱了
        adv_dir = os.path.join(config.adv_path, config.attack)
        mkdir(adv_dir)
        adv_dir = os.path.join(adv_dir, config.dataset)
        mkdir(adv_dir)
        adv_dir = os.path.join(adv_dir, str(eps))
        mkdir(adv_dir)
        # MNIST任务的预测结果为数字，方便保存init_pred和final_pred
        if config.dataset == 'MNIST':
            adv_path = os.path.join(adv_dir, f"{init_pred}->{final_pred}.png")
        elif config.dataset == 'coco':
            adv_path = os.path.join(adv_dir, f"{i}.png")
        elif config.dataset == 'CIFAR10':
            adv_path = os.path.join(adv_dir, f"{init_pred}->{final_pred}.png")
        else:
            logger.critical(f'{config.dataset} is Unknown dataset')
            exit()
        logger.debug(f'adv_dir:{adv_dir}')
        logger.debug(f'len(os.listdir(adv_dir)):{len(os.listdir(adv_dir))}')
        # 如果图片保存的不够5张
        if len(os.listdir(adv_dir)) <= 5:
            img.save(adv_path)  # 保存图片到本地，文件名包含初始预测标签和最终预测标签
            logger.info(f"Adversarial example {i} saved")
        else:
            logger.warning(f"Adversarial example {i} has been saved!")
