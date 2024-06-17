from loguru import logger

# 设置日志格式
logger.remove()  # 移除默认的日志配置
logger.add("../logs/my_log.log", format="{time:YYYY-MM-DD HH:mm:ss}|{level}|{file.path} {message}")

if __name__ == "__main__":
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("An error occurred: Division by zero")

    print("123")