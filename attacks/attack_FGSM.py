import time

import torch
from torchvision import transforms
from utils.denorm import denorm


class attack_FGSM:
    def __init__(self, model, val_DataLoader, config):
        self.model = model
        self.device = config.device
        self.val_DataLoader = val_DataLoader

    # 用于测试FGSM方法根据不同扰动对准确率的影响
    def test(self, eps):
        accuracy = 0
        adv_examples = []
        start_time = time.time()
        for img, label in self.val_DataLoader:
            img, label = img.to(self.device), label.to(self.device)
            img.requires_grad = True
            output = self.model(img)

            init_pred = output.argmax(dim=1, keepdim=True)
            # 如果已经预测错误了，就不用进行后续操作了，进行下一轮循环
            if init_pred.item() != label.item():
                continue
            loss = torch.nn.functional.nll_loss(output, label)

            self.model.zero_grad()
            loss.backward()

            # 收集图片梯度
            data_grad = img.grad.data
            # 恢复图片到原始尺度
            data_denorm = denorm(img, self.device)
            perturbed_data = self.attack(data_denorm, eps, data_grad)

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
            output = self.model(perturbed_data_normalized)

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

        # Calculate final accuracy for this epsilon
        final_acc = accuracy / float(len(self.val_DataLoader))
        end_time = time.time()
        print(
            f"Epsilon: {eps}\tTest Accuracy = {accuracy} / {len(self.val_DataLoader)} = {final_acc},Time = {end_time - start_time}")
        # Return the accuracy and an adversarial example
        return final_acc, adv_examples

    # 用于生成对抗样本
    def attack(self, image, epsilon, data_grad):
        """
        Perform FGSM with
        :param image: 输入图片
        :param epsilon: 𝜀超参数
        :param data_grad: 梯度
        :return:
        """
        # 获取梯度方向
        sign_data_grad = data_grad.sign()
        # 对原始图像添加扰动
        perturbed_image = image + epsilon * sign_data_grad
        # 将生成的对抗样本的扰动控制在0~1之间
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        return perturbed_image
