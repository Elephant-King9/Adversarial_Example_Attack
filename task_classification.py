# 用于测试FGSM方法根据不同扰动对准确率的影响
import time

from metrics.PSNR import PSNR
from utils.plot_adv_comparison import save_image_comparison
from utils.save_image import save_image
from log_config import logger


# 分类任务的攻击流程
def task_classification(eps, attacker, model, val_DataLoader, config):
    accuracy = 0
    adv_examples = []
    start_time = time.time()
    i = 0
    for data in val_DataLoader:
        i = i + 1
        img, label = data
        img, label = img.to(config.device), label.to(config.device)
        img.requires_grad = True
        output = model.predict(img)

        init_pred = output.argmax(dim=1, keepdim=True)

        if init_pred.item() != label.item():
            continue
        perturbed_data = attacker.attack(img, eps, label)
        output = model.predict(perturbed_data)
        final_pred = output.argmax(dim=1, keepdim=True)
        logger.debug(f'i:{i}, final_pred:{final_pred.item()}, label:{label.item()}, init_pred:{init_pred.item()}')

        # 保存图像
        if final_pred.item() == label.item():
            accuracy += 1
            if eps == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))
        else:
            if len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred.item(), final_pred.item(), adv_ex))

        # 调小用于测试
        if len(adv_examples) >= 5:
            break

        # 图像评估
        psnr = PSNR(img, perturbed_data)
        psnr_value = psnr.calculate_psnr()
        if psnr_value == float('inf'):
            # 代表图像完全相同
            logger.warning(f'eps:{eps}, {config.attack} attack lose efficacy')
        else:
            logger.info(f'eps:{eps}, len(adv_examples):{len(adv_examples)}, psnr_value: {psnr_value} dB')

    save_image(config, adv_examples, eps)
    save_image_comparison(config, adv_examples, eps)

    # 计算当前epsilon下的最终准确率
    final_acc = accuracy / float(len(val_DataLoader))
    end_time = time.time()
    print(
        f"Epsilon: {eps}\tTest Accuracy = {accuracy} / {len(val_DataLoader)} = {final_acc}, Time = {end_time - start_time}")
    return final_acc, adv_examples
