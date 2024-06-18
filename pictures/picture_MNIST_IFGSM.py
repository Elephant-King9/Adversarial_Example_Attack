import os

import numpy as np
from matplotlib import pyplot as plt
from log_config import logger


class Picture_MNIST_IFGSM:
    def __init__(self, config):
        # 扰动参数列表
        self.epsilons = config.epsilons
        # 正确率
        self.accuracies = config.accuracies
        # 生成的对抗样本
        self.examples = config.examples
        # 图片保存路径
        self.plt_path = config.plt_path
        self.alpha = config.alpha

    def draw(self):
        # 3.绘图,用于可视化
        # 创建一个新的图形对象，图形大小设置为 5x5 英寸
        plt.figure(figsize=(5, 5))
        # 用epsilons作为x轴数据，accuracies作为y轴数据
        # *-代表数据点用*标记，点之间用直线链接
        plt.plot(self.epsilons, self.accuracies, "*-")
        # 设置y轴刻度，
        # np.arange(0, 1.1, step=0.1)生成0~1的数组，步长为0.1
        plt.yticks(np.arange(0, 1.1, step=0.1))
        # 设置x轴刻度
        # 最大需要大一个步长
        plt.xticks(np.arange(0, 25, step=5))
        # 将图标标题设为Accuracy vs Epsilon
        plt.title(f"Accuracy vs Iters(alpha = {self.alpha})")
        # x轴标签为Epsilon
        plt.xlabel("Iters")
        # y轴标签为Accuracy
        plt.ylabel("Accuracy")

        pic_name = 'MNIST_IFSGM1.png'
        # 图片不存在的时候再保存
        if not os.path.exists(os.path.join(self.plt_path, pic_name)):
            plt.savefig(os.path.join(self.plt_path, pic_name))
            logger.info(f'save {pic_name} successfully')
        else:
            logger.warning(f'{pic_name} is saved')

        # 显示图表
        plt.show()
        cnt = 0
        plt.figure(figsize=(8, 10))
        # 行代表不同的epsilon
        for i in range(len(self.epsilons)):
            # 列代表同一epsilon生成的图像
            for j in range(len(self.examples[i])):
                cnt += 1
                plt.subplot(len(self.epsilons), len(self.examples[0]), cnt)
                plt.xticks([], [])
                plt.yticks([], [])
                if j == 0:
                    plt.ylabel("Eps: {}".format(self.epsilons[i]), fontsize=14)
                orig, adv, ex = self.examples[i][j]
                plt.title("{} -> {}".format(orig, adv))
                plt.imshow(ex, cmap="gray")
        plt.tight_layout()

        # 保存图片
        pic_name = 'MNIST_IFSGM2.png'
        if not os.path.exists(os.path.join(self.plt_path, pic_name)):
            plt.savefig(os.path.join(self.plt_path, pic_name))
            logger.info(f'save {pic_name} successfully')
        else:
            logger.warning(f'{pic_name} is saved')

        plt.show()
