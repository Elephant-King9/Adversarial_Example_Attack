import argparse
import cv2
import torch
import os
from torchvision import models, transforms
from tqdm import tqdm
from support import RGB2Lab_t, Lab2RGB_t, light_filter, Normalize, update_paras

# 定义归一化操作
norm = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# 设置设备为GPU
device = torch.device("cuda:0")

# 设置随机种子和确定性
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True

# 定义参数解析器
parser = argparse.ArgumentParser(description='Adversarial Lightness Attack')

# 添加参数
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--segment', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.5)
parser.add_argument('--model', type=str, default="resnet50")
parser.add_argument('--input_path', type=str, default="./dataset/")
parser.add_argument('--output_path', type=str, default="./result/")
parser.add_argument('--random_init', type=bool, default=False)
parser.add_argument('--init_range', type=list, default=[0, 1])
parser.add_argument('--tau', type=float, default=-0.2)
parser.add_argument('--eta', type=float, default=0.3)

# 解析参数
args = parser.parse_args()

# 设置参数
epochs = args.epochs
batch_size = args.batch_size
image_size = args.image_size
segment = args.segment
lr = args.lr
input_path = args.input_path
output_path = args.output_path + args.model + '_' + str(segment) + '_lr_' + str(lr) + '_iter_' + str(epochs) + '/'

# 加载预训练模型并设置为评估模式
if args.model == 'resnet50':
    model = models.resnet50(pretrained=True).eval()
elif args.model == 'vgg19':
    model = models.vgg19(pretrained=True).eval()
elif args.model == 'densenet121':
    model = models.densenet121(pretrained=True).eval()
elif args.model == 'mobilenet_v2':
    model = models.mobilenet_v2(pretrained=True).eval()
model.to(device)

# 获取输入路径中的所有图像文件
image_id_list = list(filter(lambda x: '.png' in x, os.listdir(input_path)))

# 定义图像转换操作
trn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((image_size, image_size)),
])

# 定义结果目录
crit = ["total", "success", "fail"]
if not os.path.exists(output_path):
    os.makedirs(output_path)
for c in crit:
    if not os.path.exists(output_path + c + '/'):
        os.makedirs(output_path + c + '/')

# 对抗攻击循环
for k in tqdm(range(len(image_id_list))):
    if k >= 0:
        # 读取和转换原始图像
        # 读取的图像存储在image_ori变量中，图像数据格式为NumPy数组，默认颜色空间为BGR。
        image_ori = cv2.imread(input_path + image_id_list[k])
        # 指定颜色转换代码，将图像从BGR颜色空间转换为RGB颜色空间。
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)

        # 将RGB图像转换到Lab颜色空间
        # 转换为Lab颜色空间并归一化
        X_ori = (RGB2Lab_t(torch.from_numpy(image_ori).cuda() / 1.0) + 128) / 255.0
        image_ori = cv2.imread(input_path + image_id_list[k])
        image_ori = cv2.cvtColor(image_ori, cv2.COLOR_BGR2RGB)
        image_ori = trn(image_ori).unsqueeze(0).cuda()
        X_ori = X_ori.unsqueeze(0)
        X_ori = X_ori.type(torch.FloatTensor)
        best_adversary = image_ori.clone()
        best_adversary = best_adversary.cuda()
        mid_image = transforms.ToPILImage()(image_ori.squeeze(0).cpu())

        # 分离L通道（光度）和a、b通道（颜色）
        light, color = torch.split(X_ori, [1, 2], dim=1)
        light_max = torch.max(light, dim=2)[0]
        light_max = torch.max(light_max, dim=2)[0]
        light_min = torch.min(light, dim=2)[0]
        light_min = torch.min(light_min, dim=2)[0]
        color = color.cuda()
        light = light.cuda()

        # 获取图像的标签
        labels = torch.argmax(model(norm(image_ori)), dim=1)
        labels_onehot = torch.zeros(labels.size(0), 1000, device=device)
        labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
        labels_infhot = torch.zeros_like(labels_onehot).scatter_(1, labels.unsqueeze(1), float('inf'))

        # 随机初始化
        if args.random_init:
            # 代表在args中启动了参数随机初始化
            # segment为分段数量
            # 一开始随机初始化的范围为[0,1]
            Paras_light = torch.rand(batch_size, 1, segment).to(device)
            # 初始化范围为[m,n]
            # init_range[1]为n
            # init_range[0]为m
            total_range = args.init_range[1] - args.init_range[0]
            # 将Paras_light一开始从[0,1]的范围映射到[m,n]范围
            Paras_light = Paras_light * total_range + args.init_range[0]
        else:
            Paras_light = torch.ones(batch_size, 1, segment).to(device)
        Paras_light.requires_grad = True

        # 迭代进行对抗攻击
        for iteration in range(epochs):
            # 修改光度值
            X_adv_light = light_filter(light, Paras_light, segment, light_max.cuda(), light_min.cuda())
            X_adv = torch.cat((X_adv_light, color), dim=1) * 255.0
            X_adv = X_adv.squeeze(0)
            X_adv = Lab2RGB_t(X_adv - 128) / 255.0
            X_adv = X_adv.type(torch.FloatTensor)
            mid_image = transforms.ToPILImage()(X_adv)
            X_adv = X_adv.unsqueeze(0).cuda()

            # 计算对抗损失
            logits = model(norm(X_adv))
            real = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other = (logits - labels_infhot).max(1)[0]
            adv_loss = torch.clamp(real - other, min=args.tau).sum()

            # 光度分布约束损失
            paras_loss = 1 - torch.abs(Paras_light).sum() / segment
            factor = args.eta
            loss = adv_loss + factor * paras_loss
            loss.backward(retain_graph=True)

            # 更新参数
            update_paras(Paras_light, lr, batch_size)

            # 预测对抗样本的分类
            x_result = trn(mid_image).unsqueeze(0).cuda()
            predicted_classes = (model(norm(x_result))).argmax(1)
            is_adv = (predicted_classes != labels)


            # 保存对抗样本
            def save_stat(criterion):
                best_adversary[is_adv] = x_result[is_adv]
                x_np = transforms.ToPILImage()(best_adversary[0].detach().cpu())
                x_np.save(os.path.join(output_path + criterion + '/', image_id_list[k * batch_size][:-4] + '.png'))


            if is_adv:
                save_stat("success")

        # 保存原始和修改后的图像
        for j in range(batch_size):
            x_np = transforms.ToPILImage()(best_adversary[j].detach().cpu())
            if labels[j] == (model(norm(best_adversary)))[j].argmax(0):
                x_np.save(os.path.join(output_path + "fail" + '/', image_id_list[k * batch_size + j][:-4] + '.png'))
                mid_image.save(
                    os.path.join(output_path + "total" + '/', image_id_list[k * batch_size + j][:-4] + '.png'))

# 清空GPU显存
torch.cuda.empty_cache()
