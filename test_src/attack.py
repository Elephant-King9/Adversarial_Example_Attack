import os

import torch
from PIL import Image

from log_config import logger
from attacks.attack_IFGSM import attack_IFGSM
from attacks.attack_MIFGSM import attack_MIFGSM
from test_src.test_config import test_Config
from models.model_MNIST import model_MNIST

pic_path = '../results/adv_pics/FGSM/MNIST/0/0_4->4.png'
model_MNIST_path = '../assets/Pre-training_files/model_MNIST_10.pth'


config = test_Config()
# 如果预训练模型未找到 则加载文件
if os.path.exists(model_MNIST_path):
    logger.info('MNIST model loaded')
    # 加载字典形式的预训练模型，并使用GPU训练
    # 定义和导入参数这两句还得分开写
    model = model_MNIST().to(config.device)
    model.load_state_dict(torch.load(model_MNIST_path))
else:
    # 预训练模型未找到
    logger.critical('model_MNIST_path model not find')
    exit()

# print(f'model loaded: {model}')
model.eval()
# 定义转换

# 加载图像
img = Image.open(pic_path)

# 应用转换
img_tensor = config.transform(img)
attack1 = attack_IFGSM(model, config)
attack2 = attack_MIFGSM(model, config)
# 创建长度为10的全零张量
label = torch.zeros(10)

# 将下标为4的位置设置为1
label[4] = 1
# 添加维度
img_tensor = img_tensor.unsqueeze(0)  # 在第0维上添加维度
label = torch.tensor([4]).unsqueeze(0)

pic1 = attack1.attack(img_tensor, 1, label)
pic2 = attack2.attack(img_tensor, 1, label)

if pic1 == pic2:
    print('same')
else:
    print('different')
