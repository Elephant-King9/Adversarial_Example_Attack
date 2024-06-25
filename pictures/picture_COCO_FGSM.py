import os
from log_config import logger
import numpy as np
from matplotlib import pyplot as plt


class Picture_COCO_FSGM:
    def __init__(self, config):
        # 扰动参数列表
        self.epsilons = config.epsilons
        # 正确率
        self.accuracies = config.accuracies
        # 生成的对抗样本
        self.examples = config.examples
        # 图片保存路径
        self.plt_path = config.plt_path

    def draw(self):
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
        pic_name = 'COCO_FSGM1.png'
        if not os.path.exists(os.path.join(self.plt_path, pic_name)):
            plt.savefig(os.path.join(self.plt_path, pic_name))
            logger.info(f'save {pic_name} successfully')
        else:
            logger.warning(f'{pic_name} is saved')

        plt.show()
