def test_dev(model, white_model_name,attack_func, bbmodel_names, method, att_type, data_dir,eplison = np.array([0.5,51]) ,
             blur_strategy=None, step_size=5, numSP=-1,mask_att_l1=2.0,direction = None,deblurred = None,gpuid=0):

    method_name = white_model_name+"_"+white_model_name+"_"+method
    result_root ="/path of project/" 

    if method[0:5] == "mbAdv":
        pert_type = "Blur"
        step_size = int(step_size)
    else:
        pert_type = "Add"

    if not os.path.exists(result_root+"results/dev/"):
        os.mkdir(result_root+"results/dev/")

    if not os.path.exists(result_root+"results/dev/"+method_name):
        os.mkdir(result_root+"results/dev/"+method_name)

    valdir = os.path.join(data_dir)

    batch_size = 1

    print("eplison:{}".format(eplison))

    if att_type == 'TA':
        file_att_type=att_type
    else:
        file_att_type = ''

    if len(eplison)==2:
        if numSP==-1 or blur_strategy not in ["bg_obj_att","obj_att","att"]:
            file_name = 'dev/' + method_name + '/{}eplison_{}_{}'.format(file_att_type, eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}/'.format(blur_strategy)
        elif numSP==-3:
            file_name = 'dev/' + method_name + '/{}eplison_{}_{}'.format(file_att_type, eplison[0],eplison[1])+'_stepsize_{}'.format(step_size)\
                        +'_blur_strategy_{}'.format(blur_strategy)+'_mask_att_l1_{}/'.format(mask_att_l1)

        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)

    elif len(eplison)==1:

        eplison[0] = np.round(eplison[0],4)
        step_size = np.round(step_size, 4)
        file_name = 'dev/' + method_name + '/{}eplison_{}'.format(file_att_type, eplison[0])+'_stepsize_{}'.format(step_size)+'_blur_strategy_{}/'.format(blur_strategy)

        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)

    print(file_name)

    if direction is not None:
        file_name = 'dev/' + method_name + '/eplison_{}_{}'.format(eplison[0], eplison[1]) + '_stepsize_{}'.format(
            step_size) + '_direction_{}_{}'.format(direction[0],direction[1])+'_blur_strategy_{}/'.format(blur_strategy)
        if not os.path.exists(result_root+"results/" + file_name):
            os.mkdir(result_root+"results/" + file_name)



    print("savename:{}".format(file_name))


    if isinstance(eplison,np.ndarray) and len(eplison)==1:
        eplison = eplison[0]

    slt_num = 1000
    val_loader = torch.utils.data.DataLoader(
        datasets.Dev(valdir, target_file='dev_dataset.csv', transform = transforms.Compose([transforms.ToTensor()])),
        batch_size=batch_size, shuffle=False)

    success_status = np.ones([slt_num])*-1.
    success_status_fmodels = []
    fb_models = []
    checkexist = True

    if deblurred is not None:
        direct_eval = True
        file_name = file_name[:-1]+"_"+deblurred+"/"
        print(file_name)
    else:
        direct_eval = False

    for forward_model_name in bbmodel_names:
        success_status_fmodels.append(np.ones([slt_num])*-1.)
        forward_model = create_fmodel("imagenet", model_name=forward_model_name, gpu=gpuid)
        fb_models.append(forward_model)

    disp = True
    if disp:
        vis = visdom.Visdom()

    for i, (images, true_labels, target_labels, index, sample_path) in enumerate(tqdm(val_loader)):

        file_path,file_full_name = os.path.split(sample_path[0])
        image_name, ext = os.path.splitext(file_full_name)
        if deblurred[:11] == "deblurganv2":
            image_name = image_name + ".jpg"
        file_name_ = file_name + image_name
        index = index.numpy()[0]

        if os.path.exists(os.path.join(result_root+"results", file_name_+".npy")) and checkexist:

            success_status[index], original, adversarial = load_adversarial(file_name_, images)
            print(file_name_+" exists!")

            # if deblurred == "deblurgan":
            #     import visdom
            #     vis = visdom.Visdom(env='Adversarial Example Showing')
            #     vis.images(torch.from_numpy(adversarial), win='adversarial results_')

            if direct_eval:
                if success_status[index]==1:
                    predictions = model.forward(adversarial)
                    criterion1 = foolbox.criteria.Misclassification()
                    if criterion1.is_adversarial(predictions, true_labels):
                        success_status[index] = 1
                    else:
                        success_status[index] = -1

                    k = 0
                    for forward_model in fb_models:
                        predictions = forward_model.forward(adversarial)
                        criterion1 = foolbox.criteria.Misclassification()
                        if criterion1.is_adversarial(predictions, true_labels):
                            success_status_fmodels[k][index] =1
                        else:
                            success_status_fmodels[k][index] = -1
                        k+=1
            else:

                # do blackbox attack
                if success_status[index] == 1:
                    if adversarial.max() > 1:
                        adversarial = adversarial / 255

                    if len(adversarial.shape) == 3:
                        adversarial = adversarial.transpose(2, 0, 1)[np.newaxis]

                    adversarial = adversarial.astype("float32")
                    k = 0
                    for forward_model in fb_models:
                        predictions = forward_model.forward(adversarial)
                        criterion1 = foolbox.criteria.Misclassification()
                        if criterion1.is_adversarial(predictions, true_labels):
                            success_status_fmodels[k][index] =1
                        else:
                            success_status_fmodels[k][index] = -1
                        k+=1
            continue

        elif direct_eval:
            continue


        print("Processing:" + file_name_)
        # try:

        if att_type=="TA":
            label_or_target_class = target_labels.numpy()
        else:
            label_or_target_class = true_labels.numpy()

        # apply the attack
        if torch.is_tensor(images):
            images = images.numpy()  # .squeeze(0).permute(1, 2, 0).numpy()

        adversarial, success_status[index] = attack_func(model, images, label_or_target_class, pert_type,
                                                         os.path.join(valdir+"_saliency", image_name + "_saliency.jpg"),
                                                         eplison, blur_strategy,step_size,numSP=numSP,mask_att_l1=mask_att_l1,direction=direction)
        # do blackbox attack
        if success_status[index] == 1:

            if disp:
                vis.images(images)
                vis.images(adversarial)
            # predict the original label
            predictions = model.forward_one(adversarial.squeeze(0))
            adv_class = np.array([np.argmax(predictions)])
            print("adv_cls:{} org_cls:{}".format(adv_class, label_or_target_class))

            if adversarial.max() > 1:
                adversarial = adversarial/ 255
            adversarial = adversarial.astype("float32")
            k=0
            for forward_model in fb_models:
                predictions = forward_model.forward(adversarial)
                criterion1 = foolbox.criteria.Misclassification()
                if criterion1.is_adversarial(predictions, true_labels):
                    success_status_fmodels[k][index] = 1
                else:
                    success_status_fmodels[k][index] = -1
                k+=1
        store_adversarial(file_name_, images, adversarial)

        #except:
        #    continue


    np.save(result_root+"results/" + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, white_model_name, slt_num), success_status)
    k=0
    for forward_model_name in bbmodel_names:
        np.save(result_root+"results/" + file_name + "/{}_{}_succ_rate{}.npy".format(white_model_name, forward_model_name, slt_num),
            success_status_fmodels[k])
        k+=1

    print("\n", method_name, "\n")