import json

import torch
from attacks.BigAdv_lib import util
from attacks.BigAdv_lib.models import color_net
from log_config import logger


class attack_Big_Adv_cadv:
    def __init__(self, model, config):
        # colornet模型，用于图像生成
        self.model = color_net().eval()
        # 加载预训练模型
        self.model.load_state_dict(torch.load('./assets/Pre-training_files/Big_Adv/latest_net_G.pth'))
        # 分类器模型
        self.classifier = model

        self.device = config.device
        self.criterion = torch.nn.CrossEntropyLoss()
        # 停止条件
        self.threshold = 0.05
        # BigAdv参数
        self.opt = Opt()
        class_idx = json.load(open('./attacks/BigAdv_lib/CIFAR10_class_index.json'))
        self.opt.idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
        logger.debug(f'opt:{self.opt}')

    # 用于生成对抗样本
    def attack(self, image, epsilon, label, **kwargs):
        image = image.clone().detach().to(self.device)
        out_rgb = image
        # Prepare hints, mask, and get current classification
        data, target = util.get_colorization_data(image, self.opt, self.model, self.classifier)
        self.opt.target = self.opt.target if self.opt.targeted else target
        optimizer = torch.optim.Adam([data['hints'].requires_grad_(), data['mask'].requires_grad_()], lr=self.opt.lr,
                                     betas=(0.9, 0.999))

        prev_diff = 0
        for itr in range(epsilon):
            # model是攻击模型
            # classifier是分类模型
            # opt是argparse
            # data是目标图像
            out_rgb, y = util.forward(self.model, self.classifier, self.opt, data)
            # print(f'out_rgb type:{type(out_rgb)}, y type:{type(y)}')
            val, idx, labels = util.compute_class(self.opt, y)
            loss = util.compute_loss(self.opt, y, self.criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print("%.5f" % (loss.item()))

            diff = val[0] - val[1]

            if self.opt.targeted:
                if idx[0] == self.opt.target and diff > self.threshold and (diff - prev_diff).abs() < 1e-3:
                    break
            else:
                if idx[0] != self.opt.target and diff > self.threshold and (diff - prev_diff).abs() < 1e-3:
                    break
            prev_diff = diff

        return out_rgb


class Opt:
    def __init__(self):
        # Big_Adv参数
        # LAB空间中ab空间的最大值
        self.ab_max = 110.
        # 用于颜色空间的量化操作
        # 粒度越细，颜色表示越精细；粒度越粗，颜色表示越粗糙。
        # 量化前：图像使用了数百万种颜色。
        # 高粒度：图像可能使用了256种颜色。尽管颜色数量大幅减少，但视觉效果仍然保持较好。
        # 低粒度：图像可能只使用16种颜色。虽然颜色数量减少更多，但图像可能会出现明显的色带效应（color banding），即颜色过渡不再平滑，出现分块现象。
        self.ab_quant = 10.
        # 对 L 通道进行归一化处理，使其数值范围适应模型的输入要求。
        self.l_norm = 100.
        # 对 L 通道进行中心化处理，使图像亮度值围绕该中心值分布，更好地适应模型的输入要求。
        self.l_cent = 50.
        # 对掩码值进行中心化处理，以便于模型更好地利用掩码信息。
        self.mask_cent = .5
        # 定义对抗攻击的目标类别，在有目标攻击中使用，指定希望模型误分类为的类别
        self.target = 444
        # 设置用于生成对抗样本的提示数量，可能用于控制模型生成对抗样本时的提示信息量。
        self.hint = 50
        # 设置优化过程中的学习率，控制参数更新的步长。
        self.lr = 1e-4
        # 指定攻击类型（1 为有目标攻击，0 为无目标攻击）。
        self.targeted = 0
        # 设置 KMeans 聚类的簇数，用于在对抗攻击中进行图像分割或特征提取。
        self.n_clusters = 8
        # 指定要修改的图像片段数量，用于控制对抗样本中需要改变的区域或片段数量。
        self.k = 4
