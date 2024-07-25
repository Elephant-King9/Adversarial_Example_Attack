
import torch
from attacks.BigAdv_lib import util
from attacks.BigAdv_lib.models import color_net


class attack_FGSM:
    def __init__(self, model, config):
        self.model = model
        self.device = config.device
        # 攻击模型
        self.adv_model = color_net().to(self.device).eval()
        # Big_Adv参数
        # LAB空间中ab空间的最大值
        ab_max = 110.
        # 用于颜色空间的量化操作
        ab_quant = 10.
        # 对 L 通道进行归一化处理，使其数值范围适应模型的输入要求。
        l_norm = 100.
        # 对 L 通道进行中心化处理，使图像亮度值围绕该中心值分布，更好地适应模型的输入要求。
        l_cent = 50.
        # 对掩码值进行中心化处理，以便于模型更好地利用掩码信息。
        mask_cent = .5
        # 定义对抗攻击的目标类别，在有目标攻击中使用，指定希望模型误分类为的类别
        target = 444
        # 设置用于生成对抗样本的提示数量，可能用于控制模型生成对抗样本时的提示信息量。
        hint = 50
        # 设置优化过程中的学习率，控制参数更新的步长。
        lr = 1e-4
        # 指定攻击类型（1 为有目标攻击，0 为无目标攻击）。
        targeted = 1
        # 设置 KMeans 聚类的簇数，用于在对抗攻击中进行图像分割或特征提取。
        n_clusters = 8
        # 指定要修改的图像片段数量，用于控制对抗样本中需要改变的区域或片段数量。
        k = 4

    # 用于生成对抗样本
    def attack(self, image, epsilon, label, **kwargs):
        """

        :param image:  data
        :param epsilon:
        :param label:
        :param kwargs:
        :return:
        """

            # Prepare hints, mask, and get current classification
            data, target = util.get_colorization_data(im, opt, model, classifier)
            opt.target = opt.target if opt.targeted else target
            optimizer = torch.optim.Adam([data['hints'].requires_grad_(), data['mask'].requires_grad_()], lr=opt.lr,
                                         betas=(0.9, 0.999))

            prev_diff = 0
            for itr in range(epsilon):
                # model是攻击模型
                # classifier是分类模型
                # opt是argparse
                # data是目标图像
                out_rgb, y = util.forward(model, classifier, opt, data)
                print(f'out_rgb type:{type(out_rgb)}, y type:{type(y)}')
                val, idx, labels = util.compute_class(opt, y)
                loss = util.compute_loss(opt, y, criterion)
                print(f'[{itr + 1}/{opt.num_iter}] Loss: {loss:.3f} Labels: {labels}')
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print("%.5f" % (loss.item()))

                diff = val[0] - val[1]

                if opt.targeted:
                    if idx[0] == opt.target and diff > threshold and (diff - prev_diff).abs() < 1e-3:
                        break
                else:
                    if idx[0] != opt.target and diff > threshold and (diff - prev_diff).abs() < 1e-3:
                        break
                prev_diff = diff

            file_name = file_name[0] + '.png'

        return perturbed_image
