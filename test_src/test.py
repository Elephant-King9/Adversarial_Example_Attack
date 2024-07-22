# ------- testing on imagenet dataset -----------------------#
#  函数用于在 ImageNet 数据集上测试各种对抗攻击方法
def test_imagenet(model, model_name, attack_func, bbmodel_names, method, att_type, data_dir, eplison=np.array([0.5, 51])
                  , blur_model=None, gpuid=0):
    method_name = model_name + "_" + model_name + "_" + method
    result_root = "/path of project/"  # "/home/wangjian/tsingqguo/BlurAttack/"

    if method[0:5] == "mbAdv":
        pert_type = "Blur"
    else:
        pert_type = "Add"

    if not os.path.exists(result_root + "results/imagenet/" + method_name):
        os.mkdir(result_root + "results/imagenet/" + method_name)

    valdir = os.path.join(data_dir, 'val')
    batch_size = 1
    workers = 4
    if eplison is not None:
        file_name = 'imagenet/' + method_name + '/' + 'eplison{}'.format(eplison) + '_blur_model_{}/'.format(blur_model)
        if not os.path.exists(result_root + "results/" + file_name):
            os.mkdir(result_root + "results/" + file_name)
    else:
        file_name = 'imagenet/' + method_name + '/'

    # random select slt_num images from the whole dataset
    def slt_images(model, valdir, slt_num, slt_images_prior=None):

        if slt_images_prior is not None:
            valid_sampler = torch.utils.data.SubsetRandomSampler(slt_images_prior.tolist())
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])),
                batch_size=1, shuffle=False,
                num_workers=0, pin_memory=True, sampler=valid_sampler)
        else:
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ])),
                batch_size=1, shuffle=False,
                num_workers=4, pin_memory=True)

        slt_num_each = slt_num / 1000
        classes_num = np.zeros([1000])
        slt_images_idx = np.zeros([slt_num], dtype=int)

        k = 0
        for i, (images, labels, index, pathname) in enumerate(tqdm(val_loader)):
            label_ = labels.numpy()[0]

            if classes_num[label_] < slt_num_each:

                if slt_images_prior is not None:
                    classes_num[label_] += 1
                    slt_images_idx[k] = index.numpy()[0]
                    k += 1
                else:
                    if torch.is_tensor(images):
                        images = images.squeeze(0).permute(1, 2, 0).numpy()
                    predictions = model.predictions(images)
                    criterion1 = foolbox.criteria.Misclassification()
                    is_adversarials = criterion1.is_adversarial(predictions, labels)

                    if not is_adversarials:
                        classes_num[label_] += 1
                        slt_images_idx[k] = index.numpy()[0]
                        k += 1

        return slt_images_idx

    slt_num = 1000
    slt_name = result_root + "results/imagenet_slt_" + str(slt_num) + ".npy"

    slt_num_saved = 5000
    slt_name_saved = result_root + "results/imagenet_slt_" + str(slt_num_saved) + ".npy"

    if os.path.exists(slt_name):
        sltIdx = np.load(slt_name)
        sltIdx.sort(axis=0)
        # if slt
    else:
        # slt images from slt_name_saved
        if os.path.exists(slt_name_saved) and slt_num_saved >= slt_num:
            sltIdx_saved = np.load(slt_name_saved)
            sltIdx = slt_images(model, valdir, slt_num, sltIdx_saved)
            sltIdx.sort(axis=0)
            np.save(slt_name, sltIdx)
        else:
            # slt images from imagenet
            sltIdx = slt_images(model, valdir, slt_num)
            # sltIdx = np.random.choice(50000,slt_num,replace=False)
            sltIdx.sort(axis=0)
            np.save(slt_name, sltIdx)

    valid_sampler = torch.utils.data.SubsetRandomSampler(sltIdx.tolist())
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ])),
        batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, sampler=valid_sampler)

    specified_idx = -1
    success_status = np.ones([slt_num]) * -1.
    success_status_fmodels = []
    fb_models = []

    for forward_model_name in bbmodel_names:
        success_status_fmodels.append(np.ones([slt_num]) * -1.)
        forward_model = create_fmodel("imagenet", model_name=forward_model_name, gpu=gpuid)
        fb_models.append(forward_model)

    for i, (images, labels, index, sample_path) in enumerate(tqdm(val_loader)):

        file_path, file_full_name = os.path.split(sample_path[0])
        file_name_, ext = os.path.splitext(file_full_name)
        file_name_ = file_name + file_name_
        index = index.numpy()[0]
        if specified_idx >= 0:
            if i > specified_idx:
                break
            elif i != specified_idx:
                continue

        if os.path.exists(os.path.join(result_root + "results", file_name_ + ".jpg")):
            success_status[sltIdx == index], original, adversarial = load_adversarial(file_name_)
            print(file_name_ + " exists!")
            # do blackbox attack
            k = 0
            if success_status[sltIdx == index] == 1:
                if adversarial.max() > 1:
                    adversarial = adversarial.transpose(2, 0, 1) / 255
                else:
                    adversarial = adversarial.transpose(2, 0, 1)
                adversarial = adversarial.astype("float32")
                for forward_model in fb_models:
                    predictions = forward_model.forward_one(adversarial)
                    criterion1 = foolbox.criteria.Misclassification()
                    if criterion1.is_adversarial(predictions, labels):
                        success_status_fmodels[k][sltIdx == index] = 1
                    else:
                        success_status_fmodels[k][sltIdx == index] = 0
            continue

        print("Processing:" + file_name_)
        try:

            if att_type is "TA":
                np.random.seed(labels + 2)
                target_class = int(np.random.random() * 1000)
                label_or_target_class = target_class
            elif att_type is "UA":
                label_or_target_class = labels.numpy()

            # apply the attack
            if torch.is_tensor(images):
                images = images.numpy()  # .squeeze(0).permute(1, 2, 0).numpy()

            adversarial, success_status[sltIdx == index] = attack_func(model, images, label_or_target_class, pert_type,
                                                                       os.path.join(result_root + "results",
                                                                                    file_name_ + "_saliency.jpg"),
                                                                       eplison, blur_model)
            if success_status[sltIdx == index] == 1:
                if adversarial.max() > 1:
                    adversarial = adversarial.transpose(2, 0, 1) / 255
                else:
                    adversarial = adversarial.transpose(2, 0, 1)
                adversarial = adversarial.astype("float32")
                for forward_model in fb_models:
                    predictions = forward_model.forward_one(adversarial)
                    criterion1 = foolbox.criteria.Misclassification()
                    if criterion1.is_adversarial(predictions, labels):
                        success_status_fmodels[k][sltIdx == index] = 1
                    else:
                        success_status_fmodels[k][sltIdx == index] = 0

            store_adversarial(file_name_, images, adversarial)

            # do blackbox attack

        except:
            continue

    np.save(result_root + "results/" + file_name + "/{}_{}_succ_rate{}.npy".format(model_name, model_name, slt_num),
            success_status)
    k = 0
    for forward_model_name in bbmodel_names:
        np.save(result_root + "results/" + file_name + "/{}_{}_succ_rate{}.npy".format(model_name, forward_model_name,
                                                                                       slt_num),
                success_status_fmodels[k])
        k += 1

    print("\n", method_name, "\n")


def main(argv):
    opts, args = getopt.getopt(sys.argv[1:], "t:d:m:g:i:w:e:b:s:n:a:r:u:",
                               ["attack_type", "dataset", "method_name", "gpu_id", "ifdobb",
                                "white_model_name", "eplison", "blur_strategy",
                                "step_size", "numSP", "mask_att_l1", "direction", "deblurred"])
    gpu_id = 0
    method_name = "mbAdv_bim"
    ifdobb = 0
    white_model_name = "inceptionv3"
    dataset = "dev"
    attack_type = "UA"
    eplison = np.array([0.1, 15])  # np.array([0.5,10])
    direction = None  # np.array([2., 2.]) #None #np.array([0., 0.])
    blur_strategy = "whole"  # joint
    step_size = 10  # 20
    numSP = -1
    mask_att_l1 = 2.0
    deblurred = None

    for op, value in opts:
        if op == '-t' or op == '--attack_type':
            attack_type = value
        if op == '-d' or op == '--dataset':
            dataset = value
        if op == '-g' or op == '--gpu_id':
            gpu_id = value
        if op == '-m' or op == '--method':
            method_name = value
        if op == '-i' or op == '--ifdobb':
            ifdobb = value
        if op == '-w' or op == '--white_model_name':
            white_model_name = value
        if op == '-e' or op == '--eplison':
            eplison = np.array(value.split(',')).astype(float)
        if op == '-b' or op == '--blur_strategy':
            blur_strategy = value
        if op == '-s' or op == '--step_size':
            step_size = float(value)
        if op == '-n' or op == '--numsp':
            numSP = float(value)
        if op == '-a' or op == '--mask_att_l1':
            mask_att_l1 = float(value)
        if op == '-r' or op == '--direction':
            direction = np.array(value.split(',')).astype(float)
            print(direction)
        if op == '-u' or op == '--deblurred':
            if value == "None":
                deblurred = None
            else:
                deblurred = value

    if dataset == 'imagenet':
        dataset_path = "./datasets/ILSVRC2012"
    elif dataset == 'cifa10':
        dataset_path = "./datasets/cifar-10-batches-py"
    elif dataset == 'dev':
        dataset_path = "./datasets/dev/images"  # "/home/wangjian/tsingqguo/dataset/dev/images"
    elif dataset == 'mnist':
        dataset_path = "./datasets/MNIST"
    elif dataset == 'sharp':
        dataset_path = "./datasets/dev/blurred_sharp/sharp"
    elif dataset == 'real':
        dataset_path = "./datasets/dev/real"

    print('dataset path:{}'.format(dataset_path))
    print('gpu id:{}'.format(gpu_id))

    os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(gpu_id)

    if dataset == "mnist":
        from tools.spatial_transformer.data_loader import fetch_dataloader
        from tools.spatial_transformer import utils as stn_utils
        json_path = os.path.join('./tools/spatial_transformer/experiments/base_stn_model/params.json')
        assert os.path.isfile(json_path), 'No json configuration file found at {}'.format(json_path)
        params = stn_utils.Params(json_path)

        params.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.manual_seed(11052018)
        if params.device.type is 'cuda': torch.cuda.manual_seed(11052018)

    else:
        params = None

    # 加载模型
    model = create_bmodel(dataset, model_name=white_model_name, gpu=gpu_id, params=params)

    if ifdobb:
        bbmodel_names = ["inceptionresnetv2", "inceptionv3", "inceptionv4", "xception"]
        bbmodel_names.remove(white_model_name)
    else:
        bbmodel_names = None

    print("\n\nStart Test...")

    if dataset == "imagenet":
        if method_name == "bim":
            test_imagenet(model, white_model_name, run_attack_bim, bbmodel_names, method_name, attack_type,
                          dataset_path, gpuid=gpu_id)
        elif method_name == "cw":
            test_imagenet(model, white_model_name, run_attack_cw, bbmodel_names, method_name, attack_type,
                          dataset_path, gpuid=gpu_id)
        elif method_name == "gblur":
            test_imagenet(model, white_model_name, run_attack_gblur, bbmodel_names, method_name, attack_type,
                          dataset_path, gpuid=gpu_id)
        elif method_name == "fgsm":
            test_imagenet(model, white_model_name, run_attack_fgsm, bbmodel_names, method_name, attack_type,
                          dataset_path, gpuid=gpu_id)
        elif method_name == "mbAdv_bim":
            test_imagenet(model, run_attack_bim, bbmodel_names, method_name, attack_type, dataset_path,
                          eplison, blur_strategy, gpuid=gpu_id)
    elif dataset == "dev":
        if method_name == "bim":
            test_dev(model, white_model_name, run_attack_bim, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        if method_name == "mifgsm":
            step_size = 2 * eplison[0] / 10
            test_dev(model, white_model_name, run_attack_mifgsm, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        elif method_name == "cw":
            test_dev(model, white_model_name, run_attack_cw, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        elif method_name == "gblur":
            test_dev(model, white_model_name, run_attack_gblur, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        elif method_name == "dblur":
            test_dev(model, white_model_name, run_attack_dblur, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        elif method_name == "mblur":
            test_dev(model, white_model_name, run_attack_mblur, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        elif method_name == "fgsm":
            test_dev(model, white_model_name, run_attack_fgsm, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        elif method_name == "mbAdv_bim":
            test_dev(model, white_model_name, run_attack_bim, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)
        elif method_name == "mbAdv_mifgsm":
            test_dev(model, white_model_name, run_attack_mifgsm, bbmodel_names, method_name, attack_type, dataset_path,
                     eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                     deblurred=deblurred, gpuid=gpu_id)

    elif dataset == "mnist":
        if method_name == "mbAdv_mifgsm":
            test_mnist(model, white_model_name, run_attack_mifgsm, bbmodel_names, method_name, attack_type,
                       dataset_path,
                       eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                       deblurred=deblurred, gpuid=gpu_id)

    elif dataset == "real":
        if method_name == "mbAdv_mifgsm":
            test_real(model, white_model_name, run_attack_mifgsm, bbmodel_names, method_name, attack_type, dataset_path,
                      eplison, blur_strategy, step_size, numSP=numSP, mask_att_l1=mask_att_l1, direction=direction,
                      deblurred=deblurred, gpuid=gpu_id)

    sys.exit(0)
