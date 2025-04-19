from utils import utils
import hsic_utils
import torch
import torch.nn.functional as F
from copy import deepcopy
from models.discri import MinibatchDiscriminator, DGWGAN
from models.generator import Generator
from models.classify import *
from compute_ood_score import compute_ood_metrics_sota, compute_ood_metrics_dcc
from losses import NegLS_CrossEntropyLoss
import time, os
from itertools import cycle

torch.autograd.set_detect_anomaly(True)


def test(model, dataloader=None, device='cuda', mode=None, a1=0.05, a2=0.5, n_classes=1000,
               ktype='gaussian', hsic_training=True, measure='HSIC', criterion=nn.CrossEntropyLoss()):
    tf = time.time()
    model.eval()
    loss, cnt, ACC, correct_top5 = 0.0, 0, 0, 0
    with torch.no_grad():
        for i, (img, iden) in enumerate(dataloader):
            bs = img.size(0)
            iden = iden.view(-1)
            img, iden = img.to(device), iden.to(device)

            if mode == "bido":
                loss, cross_loss, out_prob, hx_l_list, hy_l_list = multilayer_bido(model, criterion, img, iden, a1, a2,
                                                                                n_classes, ktype, hsic_training, measure)
            elif mode == "vib":
                _, out_prob, mu, std = model(img)
            else:
                _, out_prob = model(img)

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()

            _, top5 = torch.topk(out_prob, 5, dim=1)
            for ind, top5pred in enumerate(top5):
                if iden[ind] in top5pred:
                    correct_top5 += 1

            cnt += bs

    return ACC * 100.0 / cnt, correct_top5 * 100.0 / cnt


def train_ft_model(main_model, aux_model, 
                    train_loader, ood_loader, test_loader,
                    criterion, optimizer, lr_scheduler,
                    n_epochs, warmup_epochs=10,
                    method='energy', lambda_oe=1e-3, lambda_aux=1e-3, defense="reg", device='cuda'):

    aux_model.eval().to(device)  # Freeze aux model
    best_model = None
    best_ACC = [0.0, 0.0]
    model = None
    test_acc = [0.0]

    for epoch in range(n_epochs):
        start_time = time.time()
        main_model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        # Dynamic weight adjustment
        if epoch < warmup_epochs:
            if defense == "reg":
                checkpoint_path = "checkpoints/target_model/target_ckp/SOTA_1000/VGG16_88.26.tar"
            elif defense == "bido":
                checkpoint_path = "./checkpoints/target_model/target_ckp/VGG16_BiDO_HSIC_VGG16_0.050&0.500_80.35.tar"
            elif defense == "vib":
                checkpoint_path = "checkpoints/target_model/target_ckp/SOTA_1000/VGG16_vib_final_0.005_73.60.tar"
            elif defense == "tl":
                checkpoint_path = "checkpoints/target_model/target_ckp/SOTA_1000/VGG16_tl_final_0.17_82.61.tar"
            elif defense == "ls":
                checkpoint_path = "checkpoints/target_model/target_ckp/SOTA_1000/VGG16_ls_final_-0.05_83.84.tar"
            checkpoint = torch.load(checkpoint_path)
            print("fix")
            fixed_state_dict = utils.fix_state_dict_keys(checkpoint['state_dict'])
            main_model.load_state_dict(fixed_state_dict, strict=True)

            current_lambda_oe = 0.0
            current_lambda_aux = 0.0
            print(f"Epoch {epoch} pass by loading pretrained model at {checkpoint_path}")
            continue
        else:
            current_lambda_oe = lambda_oe
            current_lambda_aux = lambda_aux

        print(f"Lambda_oe: {current_lambda_oe:.4f} | Lambda_aux: {current_lambda_aux:.4f}")

        combined_loader = zip(train_loader, cycle(ood_loader))  

        for batch_idx, ((x_in, y_in), (x_ood, _)) in enumerate(combined_loader):
            x_in, y_in = x_in.to(device), y_in.to(device)
            x_ood = x_ood.to(device)
            # === Main task loss ===
            if defense == "bido":
                ce_loss, cross_loss, logits, hx_l_list, hy_l_list = multilayer_bido(main_model, criterion, x_in, y_in, a1=0.05, a2=0.5,
                                                                            n_classes=1000, ktype="gaussian", hsic_training=True, measure="HSIC")
            elif defense == "vib":
                _, logits, mu, std = main_model(x_in)
                cross_loss = criterion(logits, y_in)
                info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
                ce_loss = cross_loss + 0.005 * info_loss
            else:
                outputs = main_model(x_in)
                logits = outputs[1]
                ce_loss = criterion(logits, y_in)

            # === OOD regularization ===
            if defense == "bido":
                loss, cross_loss, ood_logits, hx_l_list, hy_l_list = multilayer_bido(main_model, criterion, x_ood, y_in, a1=0.05, a2=0.5,
                                                                            n_classes=1000, ktype="gaussian", hsic_training=True, measure="HSIC")
            elif defense == "vib":
                _, ood_logits, _, _ = main_model(x_ood)
            else:
                ood_outputs = main_model(x_ood)
                ood_logits = ood_outputs[1]
            
            if lambda_oe > 0:
                if method == 'energy':
                    energy_loss = torch.logsumexp(ood_logits, dim=1).mean()
                elif method == 'conf_min':
                    probs = F.softmax(ood_logits, dim=1)
                    energy_loss = probs.max(dim=1)[0].mean()
                else:  # kl_div
                    uniform = torch.ones_like(ood_logits) / ood_logits.shape[1]
                    energy_loss = F.kl_div(F.log_softmax(ood_logits, dim=1), uniform, reduction='batchmean')
            else:
                energy_loss = 0
                
            # === Aux model supervision ===
            with torch.no_grad():
                _, aux_out = aux_model(x_ood)
            aux_loss = F.kl_div(F.log_softmax(ood_logits, dim=1), 
                               F.softmax(aux_out, dim=1), reduction='batchmean')

            # === Total loss ===
            loss = ce_loss + current_lambda_oe * energy_loss + current_lambda_aux * aux_loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Metrics
            total_loss += loss.item() * x_in.size(0)
            _, predicted = logits.max(1)
            correct += predicted.eq(y_in).sum().item()
            total += x_in.size(0)

        # Evaluation
        train_loss = total_loss / total
        train_acc = 100. * correct / total
        test_acc = test(main_model, test_loader, device, mode=defense)
        
        # Update best model
        if test_acc[0] > best_ACC[0]:
            best_ACC[0] = test_acc[0]
            best_ACC[1] = test_acc[1]
            best_model = deepcopy(main_model)
        
        model = main_model  # Keep reference to current model

        # Learning rate scheduling (if any)
        if lr_scheduler:
            lr_scheduler.step()

        # Logging
        interval = time.time() - start_time
        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(
            epoch, interval, train_loss, train_acc, test_acc[0]))
        
        # OOD metrics every 5 epochs
        if (epoch + 1) % 3 == 0:
            main_model.eval()
            print("Computing OOD metrics...")
            compute_ood_metrics_dcc(main_model, ood_loader, 30+epoch, lambda_oe, lambda_aux, f"{defense}_{method}")

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC, model, test_acc


def train_reg(loaded_args, model, criterion, optimizer, lr_schedular, trainloader, testloader, n_epochs, device='cuda'):
    best_ACC = (0.0, 0.0)

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, testloader)

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        lr_schedular.step()

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc,
                                                                                                   test_acc[0]))

        if (epoch + 1) % 5 == 0:
            model.eval()
            print("Computing OOD metrics...")
            compute_ood_metrics_sota(loaded_args, model, epoch, method="reg")

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC, model, test_acc


def train_vib(loaded_args, model, criterion, optimizer, lr_schedular, trainloader, testloader, n_epochs, beta=0.005, device='cuda'):
    best_ACC = (0.0, 0.0)

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            _, out_prob, mu, std = model(img)
            cross_loss = criterion(out_prob, iden)
            info_loss = - 0.5 * (1 + 2 * std.log() - mu.pow(2) - std.pow(2)).sum(dim=1).mean()
            loss = cross_loss + beta * info_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, testloader, mode="vib")

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        lr_schedular.step()

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc,
                                                                                                   test_acc[0]))
        if (epoch + 1) % 5 == 0:
            model.eval()
            print("Computing OOD metrics...")
            compute_ood_metrics_sota(loaded_args, model, epoch, method="vib")

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC, model, test_acc


def multilayer_bido(model, criterion, inputs, target, a1, a2, n_classes, ktype, hsic_training, measure):
    hx_l_list = []
    hy_l_list = []
    bs = inputs.size(0)
    total_loss = 0

    if hsic_training:
        hiddens, out_digit = model(inputs)
        cross_loss = criterion(out_digit, target)

        total_loss += cross_loss
        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)
        for hidden in hiddens:
            hidden = hidden.view(bs, -1)

            if measure == 'HSIC':
                hxz_l, hyz_l = hsic_utils.hsic_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=5.,
                    ktype=ktype
                )
            elif measure == 'COCO':
                hxz_l, hyz_l = hsic_utils.coco_objective(
                    hidden,
                    h_target=h_target.float(),
                    h_data=h_data,
                    sigma=5.,
                    ktype=ktype
                )

            temp_hsic = a1 * hxz_l - a2 * hyz_l
            total_loss += temp_hsic

            hx_l_list.append(round(hxz_l.item(), 5))
            hy_l_list.append(round(hyz_l.item(), 5))

    else:
        feats, out_digit = model(inputs)
        cross_loss = criterion(out_digit, target)

        total_loss += cross_loss
        h_target = utils.to_categorical(target, num_classes=n_classes).float()
        h_data = inputs.view(bs, -1)

        hxy_l = hsic_utils.coco_normalized_cca(h_data, out_digit, sigma=5., ktype=ktype)

        temp_hsic = a1 * hxy_l
        total_loss += temp_hsic

        hxz_l = hxy_l
        hyz_l = hxy_l
        hx_l_list.append(round(hxz_l.item(), 5))
        hy_l_list.append(round(hyz_l.item(), 5))

    return total_loss, cross_loss, out_digit, hx_l_list, hy_l_list


def train_with_bido(loaded_args, model, criterion, optimizer, lr_scheduler, trainloader, testloader, a1, a2, n_epochs, n_classes,
               ktype='gaussian', hsic_training=True, measure='HSIC', device='cuda'):
    best_ACC = (0.0, 0.0)
    model.train()
    end = time.time()

    for epoch in range(n_epochs):

        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            loss, cross_loss, out_digit, hx_l_list, hy_l_list = multilayer_bido(model, criterion, img, iden, a1, a2,
                                                                            n_classes, ktype, hsic_training, measure)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_digit, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, testloader, mode="bido", criterion=criterion, a1=a1, a2=a2, measure=measure, n_classes=n_classes, ktype=ktype, hsic_training=hsic_training)

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        lr_scheduler.step()

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc,
                                                                                                   test_acc[0]))
        if (epoch + 1) % 5 == 0:
            model.eval()
            print("Computing OOD metrics...")
            compute_ood_metrics_sota(loaded_args, model, epoch, method=f"bido_{ktype}")

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC, model, test_acc


def train_with_tl(loaded_args, model, criterion, optimizer, lr_schedular, trainloader, testloader, n_epochs, device='cuda'):
    best_ACC = (0.0, 0.0)

    for epoch in range(n_epochs):

        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0
        model.train()

        print_frozen_status(model)

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, testloader)

        interval = time.time() - tf
        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        lr_schedular.step()

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc,
                                                                                                   test_acc[0]))
        if (epoch + 1) % 5 == 0:
            model.eval()
            print("Computing OOD metrics...")
            compute_ood_metrics_sota(loaded_args, model, epoch, method="tl")

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC, model, test_acc


def train_with_ls(loaded_args, model, full_smooth_factor, optimizer, lr_scheduler, trainloader, testloader, n_epochs=100, device='cuda'):
    best_ACC = (0.0, 0.0)
    best_model = None

    warmup_epoches = 0.5 * n_epochs
    adjust_alpha_epoches = 0.75 * n_epochs

    for epoch in range(n_epochs):
        tf = time.time()
        ACC, cnt, loss_tot = 0, 0, 0.0

        if epoch < warmup_epoches:
            alpha = 0.0
            criterion = nn.CrossEntropyLoss().cuda()
        # elif epoch < adjust_alpha_epoches:
        #     alpha = full_smooth_factor * (epoch + 1 - warmup_epoches) / (n_epochs - warmup_epoches)
        #     criterion = NegLS_CrossEntropyLoss(alpha).cuda()
        else:
            alpha = full_smooth_factor
            criterion = NegLS_CrossEntropyLoss(full_smooth_factor).cuda()
        print(alpha, criterion)

        model.train()

        for i, (img, iden) in enumerate(trainloader):
            img, iden = img.to(device), iden.to(device)
            bs = img.size(0)
            iden = iden.view(-1)

            feats, out_prob = model(img)
            cross_loss = criterion(out_prob, iden)
            loss = cross_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            out_iden = torch.argmax(out_prob, dim=1).view(-1)
            ACC += torch.sum(iden == out_iden).item()
            loss_tot += loss.item() * bs
            cnt += bs

        train_loss, train_acc = loss_tot * 1.0 / cnt, ACC * 100.0 / cnt
        test_acc = test(model, testloader)

        interval = time.time() - tf

        lr_scheduler.step()

        if test_acc[0] > best_ACC[0]:
            best_ACC = test_acc
            best_model = deepcopy(model)

        print("Epoch:{}\tTime:{:.2f}\tTrain Loss:{:.2f}\tTrain Acc:{:.2f}\tTest Acc:{:.2f}".format(epoch, interval,
                                                                                                   train_loss,
                                                                                                   train_acc,
                                                                                                   test_acc[0]))
        if (epoch + 1) % 5 == 0:
            model.eval()
            print("Computing OOD metrics...")
            compute_ood_metrics_sota(loaded_args, model, epoch, method="ls")

    print("Best Acc:{:.2f}".format(best_ACC[0]))
    return best_model, best_ACC, model, test_acc

def freeze_parameters(model, proportion=0.5):
    all_params = list(model.named_parameters())
    total_params = sum(p.numel() for _, p in all_params)
    target_frozen = int(total_params * proportion)

    frozen_count = 0
    for name, param in all_params:
        if frozen_count >= target_frozen:
            break
            
        param.requires_grad = False
        frozen_count += param.numel()
        # print(f"Froze {name:<40} | Params: {param.numel():,}")
    
    actual_frozen = sum(p.numel() for _, p in all_params if not p.requires_grad)
    actual_proportion = actual_frozen / total_params
    print(f"Actually frozen: {actual_frozen:,}/{total_params:,} ({actual_proportion*100:.1f}%)")


def get_T(model_name_T, cfg, defense):
    if model_name_T.startswith("VGG16"):
        if "vib" in defense:
            print("vib VGG16")
            T = VGG16_vib(cfg['dataset']["n_classes"])
        elif "bido" in defense:
            print("bido VGG16")
            T = VGG16_bido(cfg['dataset']["n_classes"])
        else:
            print("normal VGG16")
            T = VGG16(cfg['dataset']["n_classes"])
    elif model_name_T.startswith('IR152'):
        T = IR152(cfg['dataset']["n_classes"])
    elif model_name_T == "FaceNet64":
        T = FaceNet64(cfg['dataset']["n_classes"])
    T = torch.nn.DataParallel(T).cuda()
    ckp_T = torch.load(cfg[cfg['dataset']['model_name']]['cls_ckpts'])
    T.load_state_dict(ckp_T['state_dict'], strict=False)

    return T


def train_specific_gan(cfg, defense_name="None", mode="gan"):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name_T = cfg['dataset']['model_name']
    batch_size = cfg[model_name_T]['batch_size']
    z_dim = cfg[model_name_T]['z_dim']
    n_critic = cfg[model_name_T]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Create save folders
    root_path = cfg["root_path"]
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, os.path.join(model_name_T, defense_name)))

    print(save_model_dir)

    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # Log file
    log_path = os.path.join(save_model_dir, "attack_logs")
    os.makedirs(log_path, exist_ok=True)
    # log_file = "improvedGAN_{}.txt".format(model_name_T)
    # utils.Tee(os.path.join(log_path, log_file), 'w')
    # writer = utils.SummaryWriter(log_path)

    # Load target model
    T = get_T(model_name_T=model_name_T, cfg=cfg, defense=defense_name)
    T.eval()

    # Dataset
    dataset, dataloader = utils.init_dataloader(cfg, file_path, cfg[model_name_T]['batch_size'], mode="gan")

    # Start Training
    print("Training GAN for %s" % model_name_T)
    utils.print_params(cfg["dataset"], cfg[model_name_T])

    G = Generator(cfg[model_name_T]['z_dim'])
    DG = MinibatchDiscriminator()

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=cfg[model_name_T]['lr'], betas=(0.5, 0.999))

    entropy = utils.HLoss()

    step = 0
    for epoch in range(cfg[model_name_T]['epochs']):
        start = time.time()
        _, unlabel_loader1 = utils.init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)
        _, unlabel_loader2 = utils.init_dataloader(cfg, file_path, batch_size, mode="gan", iterator=True)

        for i, imgs in enumerate(dataloader):
            current_iter = epoch * len(dataloader) + i + 1

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)
            x_unlabel = next(unlabel_loader1) 
            x_unlabel2 = next(unlabel_loader2) 

            utils.toogle_grad(G, False)
            utils.toogle_grad(DG, True)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            # y_prob = T(imgs)[-1]
            # y = torch.argmax(y_prob, dim=1).view(-1)
            y_prob = T(imgs)
            y_prob = y_prob[1]
            y = torch.argmax(y_prob, dim=1)

            _, output_label = DG(imgs)
            _, output_unlabel = DG(x_unlabel)
            _, output_fake = DG(f_imgs)

            loss_lab = utils.softXEnt(output_label, y_prob)
            loss_unlab = 0.5 * (torch.mean(F.softplus(utils.log_sum_exp(output_unlabel))) - torch.mean(
                utils.log_sum_exp(output_unlabel)) + torch.mean(F.softplus(utils.log_sum_exp(output_fake))))
            dg_loss = loss_lab + loss_unlab

            acc = torch.mean((output_label.max(1)[1] == y).float())

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G
            if step % n_critic == 0:
                utils.toogle_grad(DG, False)
                utils.toogle_grad(G, True)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                mom_gen, output_fake = DG(f_imgs)
                mom_unlabel, _ = DG(x_unlabel2)

                mom_gen = torch.mean(mom_gen, dim=0)
                mom_unlabel = torch.mean(mom_unlabel, dim=0)

                Hloss = entropy(output_fake)
                g_loss = torch.mean((mom_gen - mom_unlabel).abs()) + 1e-4 * Hloss

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \tTime:%.2f\tG_loss:%.2f\t train_acc:%.2f" % (epoch, interval, g_loss, acc))

        torch.save({'state_dict': G.state_dict()},
                   os.path.join(save_model_dir, "improved_{}_G.tar".format(dataset_name)))
        torch.save({'state_dict': DG.state_dict()},
                   os.path.join(save_model_dir, "improved_{}_D.tar".format(dataset_name)))

        # if (epoch + 1) % 10 == 0:
        #     z = torch.randn(32, z_dim).cuda()
        #     fake_image = G(z)
        #     utils.save_tensor_images(fake_image.detach(),
        #                        os.path.join(save_img_dir, "improved_celeba_img_{}.png".format(epoch)), nrow=8)


def train_general_gan(cfg):
    # Hyperparams
    file_path = cfg['dataset']['gan_file_path']
    model_name = cfg['dataset']['model_name']
    lr = cfg[model_name]['lr']
    batch_size = cfg[model_name]['batch_size']
    z_dim = cfg[model_name]['z_dim']
    epochs = cfg[model_name]['epochs']
    n_critic = cfg[model_name]['n_critic']
    dataset_name = cfg['dataset']['name']

    # Create save folders
    root_path = cfg["root_path"]
    save_model_dir = os.path.join(root_path, os.path.join(dataset_name, 'general_GAN'))
    save_img_dir = os.path.join(save_model_dir, "imgs")
    os.makedirs(save_model_dir, exist_ok=True)
    os.makedirs(save_img_dir, exist_ok=True)

    # Log file
    log_path = os.path.join(save_model_dir, "logs")
    os.makedirs(log_path, exist_ok=True)
    log_file = "GAN_{}.txt".format(dataset_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')
    # writer = SummaryWriter(log_path)

    # Dataset
    dataset, dataloader = utils.init_dataloader(cfg, file_path, batch_size, mode="gan")

    # Start Training
    print("Training general GAN for %s" % dataset_name)
    utils.print_params(cfg["dataset"], cfg[model_name])

    G = Generator(z_dim)
    DG = DGWGAN(3)

    G = torch.nn.DataParallel(G).cuda()
    DG = torch.nn.DataParallel(DG).cuda()

    dg_optimizer = torch.optim.Adam(DG.parameters(), lr=lr, betas=(0.5, 0.999))
    g_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))

    step = 0

    for epoch in range(epochs):
        start = time.time()
        for i, imgs in enumerate(dataloader):

            step += 1
            imgs = imgs.cuda()
            bs = imgs.size(0)

            utils.toogle_grad(G, False)
            utils.toogle_grad(DG, True)

            z = torch.randn(bs, z_dim).cuda()
            f_imgs = G(z)

            r_logit = DG(imgs)
            f_logit = DG(f_imgs)

            wd = r_logit.mean() - f_logit.mean()  # Wasserstein-1 Distance
            gp = utils.gradient_penalty(imgs.data, f_imgs.data, DG=DG)
            dg_loss = - wd + gp * 10.0

            dg_optimizer.zero_grad()
            dg_loss.backward()
            dg_optimizer.step()

            # train G

            if step % n_critic == 0:
                utils.toogle_grad(DG, False)
                utils.toogle_grad(G, True)
                z = torch.randn(bs, z_dim).cuda()
                f_imgs = G(z)
                logit_dg = DG(f_imgs)
                # calculate g_loss
                g_loss = - logit_dg.mean()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

        end = time.time()
        interval = end - start

        print("Epoch:%d \t Time:%.2f\t Generator loss:%.2f" % (epoch, interval, g_loss))
        if (epoch + 1) % 10 == 0:
            z = torch.randn(32, z_dim).cuda()
            fake_image = G(z)
            utils.save_tensor_images(fake_image.detach(), os.path.join(save_img_dir, "result_image_{}.png".format(epoch)),
                               nrow=8)

        torch.save({'state_dict': G.state_dict()}, os.path.join(save_model_dir, "celeba_G.tar"))
        torch.save({'state_dict': DG.state_dict()}, os.path.join(save_model_dir, "celeba_D.tar"))

def train_augmodel(cfg):
    # Hyperparams
    target_model_name = cfg['train']['target_model_name']
    student_model_name = cfg['train']['student_model_name']
    device = cfg['train']['device']
    lr = cfg['train']['lr']
    temperature = cfg['train']['temperature']
    dataset_name = cfg['dataset']['name']
    n_classes = cfg['dataset']['n_classes']
    batch_size = cfg['dataset']['batch_size']
    seed = cfg['train']['seed']
    epochs = cfg['train']['epochs']
    log_interval = cfg['train']['log_interval']

    # Create save folder
    save_dir = os.path.join(cfg['root_path'], dataset_name)
    save_dir = os.path.join(save_dir, '{}_{}_{}_{}'.format(target_model_name, student_model_name, lr, temperature))
    os.makedirs(save_dir, exist_ok=True)

    # Log file
    now = utils.datetime.now()  # current date and time
    log_file = "studentKD_logs_{}.txt".format(now.strftime("%m_%d_%Y_%H_%M_%S"))
    utils.Tee(os.path.join(save_dir, log_file), 'w')
    torch.manual_seed(seed)

    kwargs = {'batch_size': batch_size}
    if device == 'cuda':
        kwargs.update({'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True},
                      )

    # Get models
    teacher_model = utils.get_augmodel(target_model_name, n_classes, cfg['train']['target_model_ckpt'])
    model = utils.get_augmodel(student_model_name, n_classes)
    model = model.to(device)
    print('Target model {}: {} params'.format(target_model_name, utils.count_parameters(model)))
    print('Augmented model {}: {} params'.format(student_model_name, utils.count_parameters(teacher_model)))

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Get dataset
    _, dataloader_train = utils.init_dataloader(cfg, cfg['dataset']['gan_file_path'], batch_size, mode="gan")
    _, dataloader_test = utils.init_dataloader(cfg, cfg['dataset']['test_file_path'], batch_size, mode="test")

    # Start training
    top1, top5 = test(teacher_model, dataloader=dataloader_test)
    print("Target model {}: top 1 = {}, top 5 = {}".format(target_model_name, top1, top5))

    loss_function = nn.KLDivLoss(reduction='sum')
    teacher_model.eval()
    for epoch in range(1, epochs + 1):
        model.train()
        for batch_idx, data in enumerate(dataloader_train):
            data = data.to(device)

            curr_batch_size = len(data)
            optimizer.zero_grad()
            _, output_t = teacher_model(data)
            _, output = model(data)

            loss = loss_function(
                F.log_softmax(output / temperature, dim=-1),
                F.softmax(output_t / temperature, dim=-1)
            ) / (temperature * temperature) / curr_batch_size

            loss.backward()
            optimizer.step()

            if (log_interval > 0) and (batch_idx % log_interval == 0):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(dataloader_train.dataset),
                           100. * batch_idx / len(dataloader_train), loss.item()))

        scheduler.step()
        top1, top5 = test(model, dataloader=dataloader_test)
        print("epoch {}: top 1 = {}, top 5 = {}".format(epoch, top1, top5))

        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(save_dir,
                                     "{}_{}_kd_{}_{}.pt".format(target_model_name, student_model_name, seed, epoch + 1))
            torch.save({'state_dict': model.state_dict()}, save_path)

def print_frozen_status(model):
    print("\n===== Check freezing status =====")
    total_params = 0
    frozen_params = 0
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        if not param.requires_grad:
            frozen_params += param.numel()
    
    print(f"Total params: {total_params:,} | Frozen params: {frozen_params:,} ({frozen_params/total_params:.1%})")
