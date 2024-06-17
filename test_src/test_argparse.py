import argparse
"""
创建ArgumentParser，用于命令行
description用于当用户使用-h时显示在首行

用户请求帮助：
python example.py -h

输出：
usage: example.py [-h] -n NUMBER [-v]
示例程序：解析命令行参数


"""
parser = argparse.ArgumentParser(description='示例程序：解析命令行参数')
"""
添加命令行参数
-m              为短名称
--model         为长名称
required=True   代表这个参数是必须的
choices         代表参数只能在这些里面选
help            用于当用户输入帮助时显示
"""
parser.add_argument('-m', '--model', required=True, choices=['vgg', 'inception', 'resnet'], help='选择模型')
parser.add_argument('-d','--dataset', required=True, choices=['MNIST', 'CIFAR10'], help='选择数据集')

# 进行参数解析
args = parser.parse_args()


def main():
    model = args.model
    dataset = args.dataset
    if model == 'vgg':
        print('vgg')
    elif model == 'inception':
        print('inception')
    elif model == 'resnet':
        print('resnet')
    else:
        print(f'Model {model} not recognized')
        return

    if dataset == 'MNIST':
        print('MNIST')
    elif dataset == 'CIFAR10':
        print('CIFAR10')
    else:
        print(f'Dataset {dataset} not recognized')
        return


if __name__ == '__main__':
    main()