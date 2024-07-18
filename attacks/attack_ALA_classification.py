import argparse
import cv2
import torch
import os
from torchvision import models, transforms
from tqdm import tqdm
from attacks.ALA_lib import RGB2Lab_t, Lab2RGB_t, light_filter, Normalize, update_paras
from log_config import logger


class attack_ALA_classification:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.batch_size = config.batch_size
        self.segment = config.segment
        self.init_range = config.init_range
        self.transform = config.transform
        self.tau = config.tau
        self.eta = config.eta
        self.lr = config.lr

    def attack(self, image, epsilon, label, **kwargs):
        # logger.debug(f'before attack image shape: {image.shape}')

        # 从[1, 3, 32. 32] tensor -> [32, 32, 3] tensor
        perturbed_image = image.clone().detach().to(self.device)
        perturbed_image = perturbed_image.squeeze(0)
        perturbed_image = perturbed_image.permute(1, 2, 0)
        # logger.debug(f'before attack  perturbed_image shape: {perturbed_image.shape}, type:{type(perturbed_image)}')
        # 将输出图像从[0,1]转化为[0,255]
        perturbed_image = perturbed_image * 255

        # 将RGB图像转换到Lab颜色空间并归一化
        X_ori = (RGB2Lab_t(perturbed_image / 1.0) + 128) / 255.0
        # 在第一个位置添加一个维度 [L, A, B]->[1, L, A, B]
        X_ori = X_ori.unsqueeze(0).type(torch.FloatTensor).to(self.device)
        best_adversary = perturbed_image.clone().to(self.device)
        # 将张量转换为 PIL 图像，方便后续的可视化或保存操作
        # mid_image = transforms.ToPILImage()(perturbed_image.squeeze(0).cpu())

        # 分离L通道（光度）和a、b通道（颜色）
        # light：形状为 [batch_size, 1, H, W]，包含光度L通道。
        # color：形状为 [batch_size, 2, H, W]，包含颜色a和b通道。
        light, color = torch.split(X_ori, [1, 2], dim=1)
        # light_max：形状为 [batch_size, 1]，包含每个批次图像的光度L通道的最大值。
        light_max = torch.max(light, dim=2)[0].max(dim=2)[0]
        # light_min：形状为 [batch_size, 1]，包含每个批次图像的光度L通道的最小值。
        light_min = torch.min(light, dim=2)[0].min(dim=2)[0]

        color = color.to(self.device)
        light = light.to(self.device)

        # 随机初始化
        if self.config.random_init:
            # 代表在args中启动了参数随机初始化
            # segment为分段数量
            # 一开始随机初始化的范围为[0,1]
            Paras_light = torch.rand(self.batch_size, 1, self.segment).to(self.device)
            # 初始化范围为[m,n]
            # init_range[1]为n
            # init_range[0]为m
            total_range = self.init_range[1] - self.init_range[0]
            # 将Paras_light一开始从[0,1]的范围映射到[m,n]范围
            Paras_light = Paras_light * total_range + self.init_range[0]
        else:
            Paras_light = torch.ones(self.batch_size, 1, self.segment).to(self.device)
        Paras_light.requires_grad = True

        # 迭代进行对抗攻击
        for _ in range(epsilon):
            # 修改光度值
            X_adv_light = light_filter(light, Paras_light, self.segment, light_max.to(self.device),
                                       light_min.to(self.device))
            # 将亮度拼接，重新变成LAB图像
            X_adv = torch.cat((X_adv_light, color), dim=1) * 255.0
            # 形状为 [1, C, H, W] 变为 [C, H, W]
            X_adv = X_adv.squeeze(0)
            # Lab2RGB_t(X_adv - 128)：将Lab图像转换回RGB颜色空间。在转换之前，将Lab图像值减去128，以恢复到Lab颜色空间的原始范围。
            X_adv = Lab2RGB_t(X_adv - 128) / 255.0
            X_adv = X_adv.type(torch.FloatTensor).to(self.device)
            # 将张量图像转换为PIL图像，以便于后续的图像处理或可视化。
            mid_image = transforms.ToPILImage()(X_adv)
            # 应用预定义的图像转换（例如调整大小、归一化）将PIL图像转换为张量。
            X_adv = self.transform(mid_image).unsqueeze(0).to(self.device)

            # 计算对抗损失
            # 也就是公式中的Lc&w
            # 删除归一化
            # logits = self.model(self.norm(X_adv))
            logits = self.model.predict(X_adv)
            # 获取真实类别的得分
            real = logits.gather(1, label.unsqueeze(1)).squeeze(1)
            # logger.debug(f'logits shape: {logits.shape}, real shape: {real.shape}')
            # 除真实类别外的最高得分
            other = (logits - torch.zeros_like(logits).scatter_(1, label.unsqueeze(1), float('inf'))).max(1)[0]
            adv_loss = torch.clamp(real - other, min=self.tau).sum()

            # 光度分布约束损失
            # 公式中的正则化项
            paras_loss = 1 - torch.abs(Paras_light).sum() / self.segment
            # 正则化项权重𝛽
            factor = self.eta
            loss = adv_loss + factor * paras_loss
            loss.backward(retain_graph=True)

            """
            adv_loss: 对抗损失 目标是使对抗样本的预测结果偏离真实标签，也就类似于CW中两个标签的差值
            paras_loss: 参数损失 也就是正则化项
            loss: 总损失
            """


            # 更新参数
            update_paras(Paras_light, self.lr, self.batch_size)

            # 预测对抗样本的分类
            x_result = X_adv.detach().clone()
            # 获取样本分类
            predicted_classes = self.model.predict(x_result).argmax(1)
            # 布尔值，判断是否被错误分类
            # True代表错误分类
            is_adv = (predicted_classes != label)
        if epsilon != 0:
            return x_result
        else:
            perturbed_image = perturbed_image.permute(2, 0, 1)
            perturbed_image = perturbed_image.unsqueeze(0)
            return perturbed_image
