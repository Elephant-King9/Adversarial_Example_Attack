from loguru import logger
from notifiers import get_notifier

# 配置 Loguru 的显示方式
logger.add(
    sink=lambda record: print(
        f"{record.time} | {record.level.name} | {record.name}:{record.function}:{record.line} - {record.message}"
    ),
    level="ERROR"
)

# 获取电子邮件通知器
notifier = get_notifier("email")

# 配置电子邮件通知参数
email_params = {
    "username": "project9769@163.com",
    "password": "PEUEGVQUCPUZARSD",
    "from": "project9769@163.com",
    "to": "project9769@163.com",
    "host": "smtp.163.com",
    "port": 465,
    "tls": False,
    "ssl": True,  # 使用 SSL 加密连接
    "subject": "Loguru Error Notification"
}


# 定义一个函数，用于将错误日志发送到电子邮件
def send_email_notification(message):
    # 将错误日志消息作为邮件内容
    email_params["message"] = message
    # 返回值是响应信息
    response = notifier.notify(**email_params)
    # 打印相应信息
    logger.debug(f"Email send response: {response}")


# 配置 Loguru，将错误信息发送到电子邮件
def configure_logger():
    # 在日志中记录ERROR时，将触发 send_email_notification 函数来发送邮件通知。
    logger.add(send_email_notification, level="ERROR", format="{time} | {level} | {name}:{function}:{line} - {message}")


if __name__ == "__main__":
    configure_logger()
    try:
        1 / 0
    except ZeroDivisionError:
        logger.error("An error occurred: Division by zero")
