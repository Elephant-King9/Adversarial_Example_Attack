# 选择模型，配置相应的参数
import argparse

import torch
import torchvision
from torchvision import transforms

from contrast import main

# 创建ArgumentParser，用于命令行
parser = argparse.ArgumentParser(description='select model dataset attack')

parser.add_argument('-m', '--model', type=str, required=True, choices=['MNIST'], help='model type')
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['MNIST'], help='dataset type')
parser.add_argument('-a', '--attack', type=str, required=True, choices=['FGSM', 'IFGSM'], help='attack type')

# 进行参数解析
args = parser.parse_args()


class Config:
    # 训练设备

    # NVIDIA
    train_gpu = '1'
    device = torch.device('cuda:' + train_gpu if torch.cuda.is_available() else 'cpu')

    # Mac M1
    # batch_size 为 1 的时候gpu比cpu更慢了
    # device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    # 命令行参数选择
    # 模型选择
    model = args.model
    # 数据集选择
    dataset = args.dataset
    # 攻击方式选择
    attack = args.attack

    # 数据集相关
    # 数据集是否下载
    download = False
    # transform
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # DataLoader相关
    batch_size = 1
    shuffle = True

    # 扰动参数
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    # 记录不同扰动下的准确度
    accuracies = []
    # 记录样本
    examples = []

    # plt生成图像保存的文件夹路径
    plt_path = 'results/plt_pics'
    # 显示参数
    def display(self):
        print('------------Train Device------------')
        print(f'device: {self.device}')
        print('------------Train Model------------')
        print(f'model: {self.model}')
        print('------------Attack Model------------')
        print(f'attack: {self.attack}')
        print('------------Dataset------------')
        print(f'dataset: {self.dataset}')
        print(f'download: {self.download}')
        print(f'transform: {self.transform}')
        print('------------DataLoader------------')
        print(f'batch_size: {self.batch_size}')
        print(f'shuffle: {self.shuffle}')


if __name__ == '__main__':
    config = Config()
    config.display()
    # 调用contrast.py中的main函数
    main(config)
