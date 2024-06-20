import os

from loguru import logger

# log文件夹路径
log_dir = "./logs"
# log文件路径
log_file = "log"

if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_path = os.path.join(log_dir, log_file)

logger.add(log_path)