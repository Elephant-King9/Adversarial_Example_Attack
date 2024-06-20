import os

from utils.mkdir import mkdir
from log_config import logger


# 判断设备并返回相应路径
def judge_device(device, save_path):
    device = str(device)
    if device[:4] == 'cuda':
        # 代表NVIDIA显卡
        save_path = os.path.join(save_path, 'NVIDIA')
        mkdir(save_path)
    elif device[:3] == 'mps':
        # 代表mps
        save_path = os.path.join(save_path, 'MAC')
        mkdir(save_path)
    elif device[:3] == 'cpu':
        # 代表CPU
        save_path = os.path.join(save_path, 'CPU')
        mkdir(save_path)
    else:
        logger.warning('Unsupported device')
        exit()
    return save_path
