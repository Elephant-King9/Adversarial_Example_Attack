# 选择模型，配置相应的参数
import argparse

import torch
import torchvision
from torchvision import transforms

from contrast import main
from networks.blip.blip import init_tokenizer

# 创建ArgumentParser，用于命令行
parser = argparse.ArgumentParser(description='select model dataset attack')

parser.add_argument('-m', '--model', type=str, required=True, choices=['MNIST', 'blip_caption', 'ResNet50', 'CIFAR10'], help='model type')
parser.add_argument('-d', '--dataset', type=str, required=True, choices=['MNIST', 'coco', 'CIFAR10'], help='dataset type')
parser.add_argument('-a', '--attack', type=str, required=True, choices=['FGSM', 'IFGSM', 'MIFGSM', 'gaussian_noise',
                                                                        'shot_noise', 'impulse_noise', 'speckle_noise',
                                                                        'gaussian_blur',
                                                                        'defocus_blur', 'zoom_blur', 'fog', 'frost',
                                                                        'snow', 'spatter',
                                                                        'contrast', 'brightness', 'saturate',
                                                                        'pixelate', 'elastic',
                                                                        'glass_blur', 'motion_blur', 'PGD',
                                                                        'CW_classification',
                                                                        'CW_caption', 'ALA_classification'], help='attack type')

# 进行参数解析
args = parser.parse_args()


class Config:
    # 训练设备

    # NVIDIA
    train_gpu = '4'
    device = torch.device('cuda:' + train_gpu if torch.cuda.is_available() else 'cpu')

    # Mac M1
    # batch_size 为 1 的时候gpu比cpu更慢了
    # FGSM:gpu比cpu慢
    # IFGSM:gpu比cpu慢
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
    download = True
    # transform
    transform = torchvision.transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # DataLoader相关
    batch_size = 1
    shuffle = True

    # FGSM中代表扰动参数
    # IFGSM中代表迭代轮数
    # MIFGSM中代表迭代轮数
    # PGD中代表迭代轮数
    epsilons = [0, .05, .1, .15, .2, .25, .3]

    # 记录不同扰动下的结果
    accuracies = []
    # 记录样本
    examples = []

    # plt生成图像保存的文件夹路径
    plt_path = 'results/plt_pics'
    # adv_pics图像保存路劲
    adv_path = 'results/adv_pics'
    # 预训练文件保存路径
    pre_train_path = 'assets/Pre-training_files'
    # IFGSM所需的参数
    # MIFGSM所需的参数
    # PGD所需的参数
    # 迭代步长
    alpha = 1 / 75

    # MIFGSM所需的参数
    # 动量
    momentum = 0.9

    # BLIP所需的模型
    # 图像的输入尺寸
    blip_image_size = 480

    # PGD中代表邻域
    eps = 0.3

    # CW_classification的参数
    # 优化器的学习率
    LEARNING_RATE = 1e-2
    # 置信度,kappa,用于计算损失的临界点，用于标签相关
    CONFIDENCE = 0
    # 二分查找步数，用于更新const
    BINARY_SEARCH_STEPS = 9
    # 是否提前终止，True代表开启
    ABORT_EARLY = False
    # 初始的常数const,平衡loss1和loss2
    INITIAL_CONST = 1e-3
    # 是否进行目标攻击
    TARGETED = False

    # CW_caption的参数
    # 分词器，使用blip.py中定义的分词器
    tokenizer = init_tokenizer()

    # ALA参数
    # tau 用于控制对抗损失中的阈值。
    # 当计算对抗损失时，如果真实类别的得分减去其他类别的最高得分低于 tau，
    # 则将其设置为 tau。这样可以防止损失过小，从而增强攻击效果。
    tau = -0.2
    # 𝛽
    eta = 0.3
    # [m,n]
    init_range = [0, 1]
    # 是否随机初始化
    random_init = True
    # T 分段数目
    segment = 64
    # 学习率
    lr = 0.5

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
        print('------------Save Path------------')
        print(f'adv_path: {self.adv_path}')
        print(f'plt_path: {self.plt_path}')
        print(f'pre_train_path:{self.pre_train_path}')
        if self.attack == 'IFGSM':
            print('------------IFGSM Attack------------')
            print(f'alpha: {self.alpha}')
        if self.attack == 'MIFGSM':
            print('------------MIFGSM Attack------------')
            print(f'alpha: {self.alpha}')
            print(f'momentum: {self.momentum}')
        if self.attack == 'PGD':
            print('------------PGD Attack------------')
            print(f'eps:{self.eps}')
        if self.attack == 'CW_classification':
            print('------------CW_classification Attack------------')
            print(f'LEARNING_RATE:{self.LEARNING_RATE}')
            print(f'CONFIDENCE:{self.CONFIDENCE}')
            print(f'BINARY_SEARCH_STEPS:{self.BINARY_SEARCH_STEPS}')
            print(f'ABORT_EARLY:{self.ABORT_EARLY}')
            print(f'INITIAL_CONST:{self.INITIAL_CONST}')
            print(f'TARGETED:{self.TARGETED}')
        if self.attack == 'CW_caption':
            print('------------CW_caption Attack------------')
        if self.attack == 'ALA':
            print('------------ALA_classification Attack------------')
            print(f'tau:{self.tau}')
            print(f'eta:{self.eta}')
            print(f'init_range:{self.init_range}')
            print(f'random_init:{self.random_init}')
            print(f'segment:{self.segment}')


if __name__ == '__main__':
    config = Config()
    config.display()
    # 调用contrast.py中的main函数
    main(config)
