from metrics.PSNR import PSNR
from utils.plot_adv_comparison import save_image_comparison
from utils.save_image import save_image
from log_config import logger


# Image caption任务的攻击流程
def task_caption(eps, attacker, model, val_DataLoader, config):
    adv_examples = []
    i = 0
    for data in val_DataLoader:
        i = i + 1
        """
        四个分别为
            图像id
            图像(tensor)
            反向归一化的图像(tensor)用于攻击模型，我这里没用，我直接在attack中使用denorm进行反向归一化
            图像的描述
        """
        image_id, image, image_unnorm, annotations = (
            data[0][0],
            data[1].squeeze(0),
            data[2][0],
            data[3:],
        )
        # 生成原图的预测结果
        # 这里传入的参数annotations好像是没啥用
        init_pred = model.predict(image_id, image, annotations, display=False)
        # 生成对抗样本
        perturbed_data = attacker.attack(image, eps, annotations, image_id=image_id, init_pred=init_pred)

        # 对攻击后生成的新图像生成预测结果
        final_pred = model.predict(image_id, perturbed_data, image_unnorm, display=False)

        # 图像评估
        psnr = PSNR(image, perturbed_data)
        psnr_value = psnr.calculate_psnr()
        if psnr_value == float('inf'):
            # 代表图像完全相同
            logger.warning(f'eps:{eps}, {config.attack} attack lose efficacy')
        else:
            logger.info(f'eps:{eps}, psnr_value: {psnr_value} dB')

        if (eps == 0 or (eps != 0 and psnr_value != 0)) and len(adv_examples) < 3:
            # 攻击后的图像
            adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
            # 原图
            orig_ex = image.squeeze().detach().cpu().numpy()
            adv_examples.append((init_pred, final_pred, orig_ex, adv_ex))
        # 用于测试，先把循环搞小点
        elif len(adv_examples) >= 3:
            break

    # 将图像存储到本地
    save_image(config, adv_examples, eps)
    save_image_comparison(config, adv_examples, eps)
    return adv_examples
