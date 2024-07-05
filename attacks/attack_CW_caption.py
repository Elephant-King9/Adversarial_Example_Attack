import torch
from nltk.translate.bleu_score import sentence_bleu


class attack_CW_caption:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.c = config.c
        self.lr = config.lr

    def attack(self, image, epsilon, label):
        """
        Perform CW attack
        :param image: 输入图片
        :param epsilon: 迭代轮数
        :param label: 标签
        """
        image = image.clone().detach().to(self.config.device)
        label = label.to(self.config.device)
        perturbed_image = image.clone().detach()
        perturbed_image.requires_grad = True
        optimizer = torch.optim.Adam([perturbed_image], lr=self.lr)

        for _ in range(epsilon):
            optimizer.zero_grad()

            real_caption = self.model.predict(perturbed_image)
            target_loss = self.compute_caption_loss(real_caption, label)

            l2_loss = torch.norm(perturbed_image - image)
            loss = self.c * target_loss + l2_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                perturbed_image.data = torch.clamp(perturbed_image, 0, 1)

        return perturbed_image

    # 用BLEU计算caption的差距
    def compute_caption_loss(self, real_caption, target_caption):
        # 使用BLEU评分计算描述文本之间的损失
        reference = [target_caption.split()]
        candidate = real_caption.split()
        bleu_score = sentence_bleu(reference, candidate)
        loss = 1 - bleu_score  # BLEU分数越高越好，所以损失是1减去BLEU分数
        return torch.tensor(loss).float().to(self.device)