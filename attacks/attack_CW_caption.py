import torch
import torch.optim as optim
from torchtext.data.metrics import bleu_score
from log_config import logger


class attack_CW_captioning:
    def __init__(self, model, config, tokenizer):
        self.model = model
        self.config = config
        self.device = config.device
        self.batch_size = config.batch_size
        self.confidence = config.CONFIDENCE  # 置信度
        self.learning_rate = config.LEARNING_RATE  # 学习率
        self.binary_search_steps = config.BINARY_SEARCH_STEPS  # 二分搜索步数
        self.abort_early = config.ABORT_EARLY  # 提前终止
        self.initial_const = config.INITIAL_CONST  # 初始常数c
        self.targeted = config.TARGETED  # 是否进行目标攻击
        self.tokenizer = config.tokenizer  # 分词器

    def attack(self, image, epsilon, target_caption=None, **kwargs):
        """
        Perform CW attack on image captioning
        :param image: 输入图片
        :param target_caption: 目标字幕（如果是目标攻击）
        :param epsilon: 迭代次数
        """

        image = image.clone().detach().to(self.device)
        perturbed_image = image.clone().detach().requires_grad_(True)

        # 获取目标字幕的 token 序列
        if self.targeted and target_caption:
            target_tokens = self.tokenizer.encode(target_caption, return_tensors="pt").to(self.device)
        else:
            target_tokens = None

        # 初始化变量
        best_perturbed_image = image.clone()
        best_loss = float('inf')

        lower_bound = torch.zeros(self.batch_size).to(self.device)
        const = torch.ones(self.batch_size).to(self.device) * self.initial_const
        upper_bound = torch.ones(self.batch_size).to(self.device) * 1e10

        for binary_search_step in range(self.binary_search_steps):
            logger.debug(f'const:{const.item()}')
            optimizer = optim.Adam([perturbed_image], lr=self.learning_rate)
            prev_loss = float('inf')

            loss1 = torch.zeros(self.batch_size).to(self.device)
            loss2 = torch.zeros(self.batch_size).to(self.device)

            for iteration in range(epsilon):
                optimizer.zero_grad()

                generated_caption = self.model.predict(image_id=None, image=perturbed_image, annotations=None)

                if self.targeted and target_tokens is not None:
                    loss1 = self._caption_loss(generated_caption, target_caption)
                else:
                    original_caption = self.model.predict(image_id=None, image=image, annotations=None)
                    loss1 = -self._caption_loss(generated_caption, original_caption)

                l2_loss = torch.sum((perturbed_image - image) ** 2, dim=[1, 2, 3])
                loss2 = l2_loss
                logger.debug(f'l2_loss: {l2_loss.item()}')
                loss = torch.sum(const * loss1 + loss2)
                logger.debug(f'epslion:{epsilon}, loss1:{loss1.item()}, loss2:{loss2.item()}, Loss:{loss.item()}')

                loss.backward()
                optimizer.step()

                if self.abort_early and iteration % (epsilon // 10) == 0:
                    if loss > prev_loss * 0.9999:
                        break
                    prev_loss = loss

                if loss < best_loss and (loss1 == 0 if self.targeted else True):
                    best_loss = loss
                    best_perturbed_image = perturbed_image.clone().detach()
                    logger.debug(f'Best loss:{best_loss}')

                with torch.no_grad():
                    perturbed_image.data = torch.clamp(perturbed_image, 0, 1)

            for i in range(self.batch_size):
                if (loss1[i] == 0 if self.targeted else True) and (loss2[i] < best_loss):
                    upper_bound[i] = min(upper_bound[i], const[i])
                    if upper_bound[i] < 1e9:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = max(lower_bound[i], const[i])
                    if upper_bound[i] < 1e9:
                        const[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        const[i] *= 10
        return best_perturbed_image

    def _caption_loss(self, generated_caption, reference_caption):
        """
        Compute the loss between generated caption and reference caption
        """
        generated_tokens = self.tokenizer.tokenize(generated_caption)
        reference_tokens = [self.tokenizer.tokenize(reference_caption)]

        # 计算BLEU分数
        bleu = bleu_score([generated_tokens], reference_tokens)
        loss = 1.0 - bleu  # BLEU分数越高，损失越低
        return torch.tensor(loss, dtype=torch.float32).to(self.device)

# 用法示例（假设config已经定义）
# from transformers import BertTokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = model_blip_caption(config)  # 你的图像字幕模型
# attack = attack_CW_captioning(model, config, tokenizer)
# perturbed_image = attack.attack(image, epsilon=1000, target_caption="a dog is running")
