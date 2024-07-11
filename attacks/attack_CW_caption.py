import torch
from nltk.translate.bleu_score import sentence_bleu
from log_config import logger

class attack_CW_caption:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.device
        self.c = config.c
        self.lr = config.lr
        self.k = config.k

    def attack(self, image, epsilon, label, **kwargs):
        """
        Perform CW attack
        :param image: 输入图片
        :param epsilon: 迭代轮数
        :param label: 初试预测
        """
        image_id = kwargs.get('image_id', None)
        init_pred = kwargs.get('init_pred', None)
        annotations = kwargs.get('annotations', None)
        
        image = image.clone().detach().to(self.config.device)
        perturbed_image = image.clone().detach()
        perturbed_image.requires_grad = True
        optimizer = torch.optim.Adam([perturbed_image], lr=self.lr)

        for iteration in range(epsilon):
            optimizer.zero_grad()

            real_caption = self.model.predict(image_id, perturbed_image, annotations, display=True)
            target_loss = self.compute_caption_loss(real_caption, init_pred)

            l2_loss = torch.norm(perturbed_image - image, p=2)
            loss = self.c * target_loss + l2_loss

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                perturbed_image.data = torch.clamp(perturbed_image, 0, 1)

            # Log the loss and check the perturbation magnitude
            logger.debug(f'Iteration {iteration+1}/{epsilon}, Loss: {loss.item()}, L2 loss: {l2_loss.item()}')
            perturbation = torch.norm(perturbed_image - image).item()
            logger.debug(f'Perturbation magnitude: {perturbation}')

        return perturbed_image

    def compute_caption_loss(self, real_caption, target_caption):
        # 使用BLEU评分计算描述文本之间的损失
        reference = [target_caption.split()]
        candidate = real_caption.split()
        bleu_score = sentence_bleu(reference, candidate)
        loss = 1 - bleu_score  # BLEU分数越高越好，所以损失是1减去BLEU分数
        return torch.tensor(loss).float().to(self.device)
