# 用于测试FGSM方法根据不同扰动对准确率的影响
import os.path
import time

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from log_config import logger
from utils.mkdir import mkdir


def attack_flow(eps, attacker, model, val_DataLoader, config):
    accuracy = 0
    adv_examples = []
    start_time = time.time()
    for img, label in val_DataLoader:
        img, label = img.to(config.device), label.to(config.device)
        img.requires_grad = True
        output = model(img)

        init_pred = output.argmax(dim=1, keepdim=True)
        # 如果已经预测错误了，就不用进行后续操作了，进行下一轮循环
        # 因为要主要判断原本正确的样本，经过对抗样本攻击后受到的影响
        if init_pred.item() != label.item():
            continue

        perturbed_data = attacker.attack(img, eps, label)

        """
        重新进行归一化处理
        如果不对生成的对抗样本进行归一化处理，程序可能会受到以下几个方面的影响：

        1. 输入数据分布不一致
        模型在训练时，输入数据经过了归一化处理，使得数据的分布具有均值和标准差的特定统计特性。如果对抗样本在进行攻击后没有进行归一化处理，其数据分布将与模型训练时的数据分布不一致。这种不一致可能导致模型对对抗样本的预测不准确。

        2. 模型性能下降
        由于输入数据分布的变化，模型的权重和偏置项可能无法适应未归一化的数据，从而导致模型性能下降。模型可能无法正确分类这些未归一化的对抗样本，从而影响模型的预测准确率。

        3. 扰动效果不可控
        在 FGSM 攻击中，添加的扰动是在未归一化的数据上进行的。如果不进行归一化处理，这些扰动在模型输入阶段可能会被放大或缩小，影响攻击的效果。这样，攻击的成功率和对抗样本的生成效果可能会变得不可控。
        """
        perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
        output = model(perturbed_data_normalized)
        final_pred = output.argmax(dim=1, keepdim=True)
        if final_pred.item() == label.item():
            accuracy += 1
            if eps == 0 and len(adv_examples) < 5:
                """
                perturbed_data 是经过FGSM攻击后的对抗样本，仍是一个tensor张量
                squeeze 会移除所有大小为1的维度
                    比如MNIST中batch_size = 1 channel=1 像素为28x28，则perturbed_data.shape = (1,1,28,28)
                    通过squeeze会变为(28,28)
                detach      代表不在跟踪其梯度，类似于
                            你有一个银行账户（相当于张量 x），你希望在这个账户基础上做一些假设性的计算（比如计划未来的支出），
                            但不希望这些假设性的计算影响到实际的账户余额。
                            银行账户余额（张量 x）：

                            你现在的账户余额是 $1000。
                            你可以对这个余额进行正常的交易（如存款、取款），这些交易会影响余额。
                            假设性的计算（使用 detach()）：

                            你想做一些假设性的计算，比如计划未来的支出，看看在不同情况下余额会变成多少。
                            你将当前余额复制一份（使用 detach()），对这份复制的余额进行操作。
                            不管你对复制的余额进行什么操作，都不会影响到实际的账户余额。
                cpu 将张量从GPU移到CPU，因为NumPy不支持GPU张量
                numpy   将tensor转化为Numpy数组
                """
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))


    # 计算当前epsilon下的最终准确率
    final_acc = accuracy / float(len(val_DataLoader))
    end_time = time.time()
    print(
        f"Epsilon: {eps}\tTest Accuracy = {accuracy} / {len(val_DataLoader)} = {final_acc}, Time = {end_time - start_time}")

    # 将对抗样本保存为图片
    for i, (init_pred, final_pred, adv_ex) in enumerate(adv_examples):
        # 将图像归一化到0-255范围并转换为uint8类型
        adv_ex = (adv_ex * 255).astype(np.uint8)
        if adv_ex.shape[0] == 1:  # 如果是单通道图像，调整为(height, width)形状
            adv_ex = adv_ex[0]
        img = Image.fromarray(adv_ex)  # 使用PIL库将NumPy数组转换为图片

        # 保存对抗样本图片
        # 创建多级文件夹，防止生成结果太乱了
        adv_dir = os.path.join(config.adv_path, config.attack)
        mkdir(adv_dir)
        adv_dir = os.path.join(adv_dir, config.dataset)
        mkdir(adv_dir)
        adv_dir = os.path.join(adv_dir, str(eps))
        mkdir(adv_dir)
        adv_path = os.path.join(adv_dir, f"{init_pred}->{final_pred}.png")
        # 如果图片保存的不够5张
        if not os.path.exists(len(os.listdir(adv_dir)) <= 5):
            img.save(adv_path)  # 保存图片到本地，文件名包含初始预测标签和最终预测标签
            logger.info(f"Adversarial example {i} saved")
        else:
            logger.warning(f"Adversarial example {i} has been saved!")
    # Return the accuracy and an adversarial example
    return final_acc, adv_examples
