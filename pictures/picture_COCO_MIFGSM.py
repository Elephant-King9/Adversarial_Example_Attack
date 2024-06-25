import os

import numpy as np
from matplotlib import pyplot as plt
from log_config import logger


class Picture_COCO_MIFGSM:
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
        cnt = 0
        plt.figure(figsize=(18, 24))
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
                plt.title("{} \n-> {}".format(orig, adv), fontsize=12)
                ex = np.transpose(ex, (1, 2, 0))
                # 确保图像数据在 [0, 1] 范围内（如果是浮点数类型）
                if ex.dtype == np.float32 or ex.dtype == np.float64:
                    ex = np.clip(ex, 0, 1)
                # 确保图像数据在 [0, 255] 范围内（如果是整数类型）
                elif ex.dtype == np.uint8 or ex.dtype == np.int32 or ex.dtype == np.int64:
                    ex = np.clip(ex, 0, 255)
                plt.imshow(ex)
        plt.tight_layout()

        # 保存图片
        pic_name = 'COCO_MIFSGM1.png'
        if not os.path.exists(os.path.join(self.plt_path, pic_name)):
            plt.savefig(os.path.join(self.plt_path, pic_name))
            logger.info(f'save {pic_name} successfully')
        else:
            logger.warning(f'{pic_name} is saved')

        plt.show()
