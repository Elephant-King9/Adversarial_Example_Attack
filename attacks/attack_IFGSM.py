import torch
import time
from torchvision import transforms

class attack_IFGSM:
    def __init__(self, model, val_DataLoader, config):
        self.model = model
        self.device = config.device
        self.val_DataLoader = val_DataLoader
        self.alpha = config.alpha
        self.iters = config.iters

    # def test(self, eps):
    #     accuracy = 0
    #     adv_examples = []
    #     start_time = time.time()
    #     for img, label in self.val_DataLoader:
    #         img, label = img.to(self.device), label.to(self.device)
    #         img.requires_grad = True
    #         output = self.model(img)
    #
    #         init_pred = output.argmax(dim=1, keepdim=True)
    #         if init_pred.item() != label.item():
    #             continue
    #
    #         perturbed_data = self.attack(img, eps, label)
    #
    #         perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
    #         output = self.model(perturbed_data_normalized)
    #
    #         final_pred = output.argmax(dim=1, keepdim=True)
    #         if final_pred.item() == label.item():
    #             accuracy += 1
    #             if eps == 0 and len(adv_examples) < 5:
    #                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
    #                 adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    #         else:
    #             if len(adv_examples) < 5:
    #                 adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
    #                 adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
    #
    #     final_acc = accuracy / float(len(self.val_DataLoader))
    #     end_time = time.time()
    #     print(f"Epsilon: {eps}\tTest Accuracy = {accuracy} / {len(self.val_DataLoader)} = {final_acc}, Time = {end_time - start_time}")
    #     return final_acc, adv_examples

    import torch

    def attack(self, image, epsilon, data_grad):
        """
        Perform IFGSM attack
        :param image: 输入图片
        :param epsilon: 𝜀超参数
        :param data_grad: 梯度
        :return: 生成的对抗样本
        """
        # 克隆原始图像，以免修改原图
        perturbed_image = image.clone().detach().to(self.device)

        # 反向归一化处理
        denorm = transforms.Normalize((0.1307,), (0.3081,))  # MNIST数据集的均值和标准差
        image_denorm = denorm(image)

        for _ in range(self.iters):
            perturbed_image.requires_grad = True
            output = self.model(perturbed_image)
            loss = torch.nn.functional.nll_loss(output, label)
            self.model.zero_grad()
            loss.backward()
            data_grad = perturbed_image.grad.data
            sign_data_grad = data_grad.sign()
            perturbed_image = image_denorm + self.alpha * sign_data_grad
            # 将对抗样本的扰动限制在原始图像的epsilon范围内
            perturbed_image = torch.clamp(perturbed_image, image - epsilon, image + epsilon)
            # 将生成的对抗样本的值限制在0到1之间
            perturbed_image = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image

