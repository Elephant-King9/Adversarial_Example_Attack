# 用于新建文件夹
import os
from loguru import logger


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        logger.info(f"First_save:Created {path}")
