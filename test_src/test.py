# 用于测试FGSM方法根据不同扰动对准确率的影响
import time

import numpy as np
from matplotlib import pyplot as plt
from torchvision import transforms

from metrics.PSNR import PSNR
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
        # 代表DataLoader的返回值只用img和label，也就是MNIST数据集
        # 主要的功能就是，选择原本预测对的图片，经过不同参数的攻击后，判断输出结果
        if len(data) == 2:
            img, label = data
            img, label = img.to(config.device), label.to(config.device)
            # logger.info(f'data size = 2,img shape:{img.shape}')
            img.requires_grad = True
            output = model.predict(img)

            init_pred = output.argmax(dim=1, keepdim=True)
            # logger.debug(f'i:{i}, label:{label.item()}, init_pred:{init_pred.item()}')
            # 如果已经预测错误了，就不用进行后续操作了，进行下一轮循环
            # 因为要主要判断原本正确的样本，经过对抗样本攻击后受到的影响
            if init_pred.item() != label.item():
                continue
            perturbed_data = attacker.attack(img, eps, label)
            # logger.debug('after attack')
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
            # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)
            output = model.predict(perturbed_data)
            final_pred = output.argmax(dim=1, keepdim=True)
            logger.debug(f'i:{i}, final_pred:{final_pred.item()}, label:{label.item()}, init_pred:{init_pred.item()}')
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

        # 代表Coco数据集完成Image Caption任务
        elif len(data) == 4:
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
            logger.info(f'image shape:{image.shape}')
            # 生成原图的预测结果
            # 这里传入的参数annotations好像是没啥用
            init_pred = model.predict(image_id, image, annotations, display=True)
            # init_pred_1 = model.predict(image_id, image, annotations, display=True)
            # logger.info(f'init_pred:{init_pred}')
            # logger.info(f'init_pred_1:{init_pred_1}')

            perturbed_data = attacker.attack(image, eps, annotations, image_id=image_id, init_pred=init_pred)

            # 将攻击后生成的图像重新进行标准化
            # perturbed_data = transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
            #                                       (0.26862954, 0.26130258, 0.27577711))(perturbed_data)
            # 测试图像标准化问题，先注释掉
            # perturbed_data_normalized = transforms.Normalize((0.1307,), (0.3081,))(perturbed_data)

            # 对攻击后生成的新图像生成预测结果
            final_pred = model.predict(image_id, perturbed_data, image_unnorm, display=True)

            # 测试输出图片
            # test_image = perturbed_data.cpu().numpy()
            # test_image = test_image.squeeze(0)
            # logger.debug(f"image_shape: {perturbed_data.shape}")
            # test_image = np.transpose(test_image, (1, 2, 0))
            # plt.imshow(test_image)
            # plt.show()

            if eps == 0 and len(adv_examples) < 5:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred, final_pred, adv_ex))
            if len(adv_examples) < 3:
                adv_ex = perturbed_data.squeeze().detach().cpu().numpy()
                adv_examples.append((init_pred, final_pred, adv_ex))

            # 用于测试，先把循环搞小点
            if len(adv_examples) >= 3:
                break
            # 图像评估
            psnr = PSNR(image, perturbed_data)
            psnr_value = psnr.calculate_psnr()
            if psnr_value == float('inf'):
                # 代表图像完全相同
                logger.warning(f'eps:{eps}, {config.attack} attack lose efficacy')
            else:
                logger.info(f'eps:{eps}, psnr_value: {psnr_value} dB')

    save_image(config, adv_examples, eps)

    # 单独保存图片信息，只用MNIST数据集的时候才保存
    if config.dataset == 'MNIST':
        # 计算当前epsilon下的最终准确率
        final_acc = accuracy / float(len(val_DataLoader))
        end_time = time.time()
        print(
            f"Epsilon: {eps}\tTest Accuracy = {accuracy} / {len(val_DataLoader)} = {final_acc}, Time = {end_time - start_time}")
        # Return the accuracy and an adversarial example
        return final_acc, adv_examples
    else:
        # coco数据集就不用返回预测的准确率了,仅返回保存的样本
        return adv_examples
