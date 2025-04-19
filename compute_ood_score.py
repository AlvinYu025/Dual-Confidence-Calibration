
import seaborn as sns
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import anom_utils
from utils import utils
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from models.classify import *
from metrics.accuracy import Accuracy


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_metric(known, novel, method):
    tp, fp, fpr_at_tpr95 = get_curve(known, novel, method)
    results = dict()

    # FPR
    mtype = 'FPR'
    results[mtype] = fpr_at_tpr95

    # AUROC
    mtype = 'AUROC'
    tpr = np.concatenate([[1.], tp / tp[0], [0.]])
    fpr = np.concatenate([[1.], fp / fp[0], [0.]])
    results[mtype] = -np.trapz(1. - fpr, tpr)

    # DTERR
    mtype = 'DTERR'
    results[mtype] = ((tp[0] - tp + fp) / (tp[0] + fp[0])).min()

    # AUIN
    mtype = 'AUIN'
    denom = tp + fp
    denom[denom == 0.] = -1.
    pin_ind = np.concatenate([[True], denom > 0., [True]])
    pin = np.concatenate([[.5], tp / denom, [0.]])
    results[mtype] = -np.trapz(pin[pin_ind], tpr[pin_ind])

    # AUOUT
    mtype = 'AUOUT'
    denom = tp[0] - tp + fp[0] - fp
    denom[denom == 0.] = -1.
    pout_ind = np.concatenate([[True], denom > 0., [True]])
    pout = np.concatenate([[0.], (fp[0] - fp) / denom, [.5]])
    results[mtype] = np.trapz(pout[pout_ind], 1. - fpr[pout_ind])

    return results


def get_curve(known, novel, method):
    tp, fp = dict(), dict()
    fpr_at_tpr95 = dict()

    known.sort()
    novel.sort()

    end = np.max([np.max(known), np.max(novel)])
    start = np.min([np.min(known), np.min(novel)])

    all = np.concatenate((known, novel))
    all.sort()

    num_k = known.shape[0]
    num_n = novel.shape[0]

    if method == 'row':
        threshold = -0.5
    else:
        threshold = known[round(0.05 * num_k)]

    tp = -np.ones([num_k + num_n + 1], dtype=int)
    fp = -np.ones([num_k + num_n + 1], dtype=int)
    tp[0], fp[0] = num_k, num_n
    k, n = 0, 0
    for l in range(num_k + num_n):
        if k == num_k:
            tp[l + 1:] = tp[l]
            fp[l + 1:] = np.arange(fp[l] - 1, -1, -1)
            break
        elif n == num_n:
            tp[l + 1:] = np.arange(tp[l] - 1, -1, -1)
            fp[l + 1:] = fp[l]
            break
        else:
            if novel[n] < known[k]:
                n += 1
                tp[l + 1] = tp[l]
                fp[l + 1] = fp[l] - 1
            else:
                k += 1
                tp[l + 1] = tp[l] - 1
                fp[l + 1] = fp[l]

    j = num_k + num_n - 1
    for l in range(num_k + num_n - 1):
        if all[j] == all[j - 1]:
            tp[j] = tp[j + 1]
            fp[j] = fp[j + 1]
        j -= 1

    fpr_at_tpr95 = np.sum(novel > threshold) / float(num_n)

    return tp, fp, fpr_at_tpr95


def compute_average_results(all_results):
    mtypes = ['FPR', 'DTERR', 'AUROC', 'AUIN', 'AUOUT']
    avg_results = dict()

    for mtype in mtypes:
        avg_results[mtype] = 0.0

    for results in all_results:
        for mtype in mtypes:
            avg_results[mtype] += results[mtype]

    print("len of all results", float(len(all_results)))
    for mtype in mtypes:
        avg_results[mtype] /= float(len(all_results))

    return avg_results


def get_energy_score(model, val_loader):
    model.eval()
    init = True
    in_energy = AverageMeter()

    model.eval()
    with torch.no_grad():
        for i, (images, _) in enumerate(val_loader):
            images = images.cuda()
            _, outputs = model(images)
            e_s = -torch.logsumexp(outputs, dim=1)
            e_s = e_s.data.cpu().numpy()
            in_energy.update(e_s.mean())

            if init:
                sum_energy = e_s
                init = False
            else:
                sum_energy = np.concatenate((sum_energy, e_s))

    print('Energy Sum {in_energy.val:.4f} ({in_energy.avg:.4f})'.format(in_energy=in_energy))
    return -1 * sum_energy


def get_msp_score(model, val_loader, nclasses, device="cuda"):
    init = True
    in_energy = AverageMeter()
    model.eval()
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            if i > 48:
                break
            if images.shape[0] == 1:
                continue
            outputs = model(images)
            outputs = outputs[1]

            soft_label = F.softmax(outputs, dim=1).detach().cpu().numpy()
            in_soft_label = soft_label[:, :nclasses]
            scores = np.max(in_soft_label, axis=1)

            in_energy.update(scores.mean())

            if init:
                sum_scores = scores
                init = False
            else:
                sum_scores = np.concatenate((sum_scores, scores))

    print('Min Conf: ', np.min(sum_scores), 'Max Conf: ', np.max(sum_scores), 'Average Conf: ', np.mean(sum_scores))

    return sum_scores


def get_score(model, val_loader, mode='msp', nclasses=1000):
    if mode == "energy":
        score = get_energy_score(model, val_loader, nclasses)

    elif mode == "msp":
        score = get_msp_score(model, val_loader, nclasses)

    return score


def compute_ood_metrics_sota(loaded_args, target_model, epoch, method="reg"):
    in_dataset_file = loaded_args['dataset']['test_file_path']
    in_classes = loaded_args['dataset']['n_classes']
    out_dataset_file = loaded_args['OOD_dataset']['test_file_path']
    out_classes = loaded_args['OOD_dataset']['n_classes']

    # Initialize data loaders
    _, testloader = utils.init_dataloader(loaded_args, in_dataset_file, mode="test")
    _, testloaderOut = utils.init_dataloader(loaded_args, out_dataset_file, mode="test")

    # Define metric
    metric = Accuracy

    # Calculate test accuracy
    test_acc = dry_evaluate(target_model, testloader, metric)
    print(f"test_acc:{test_acc:.4f}")

    # OOD mode and scoring
    mode = 'msp'  # Options: 'msp', 'energy'
    ood_sum_score = get_score(target_model, testloaderOut, mode, nclasses=out_classes)
    id_sum_score = get_score(target_model, testloader, mode, nclasses=in_classes)

    # Calculate OOD metrics
    auroc, aupr, fpr = anom_utils.get_and_print_results(id_sum_score, ood_sum_score, "CelebA", "100")
    results = cal_metric(known=id_sum_score, novel=ood_sum_score, method="msp sum")
    ood = results["AUOUT"]
    id = results["AUIN"]
    print(f"The scores: OOD {ood}, ID {id}")

    # Prepare results string
    results_str = (
        f"Epoch: {epoch}\n"
        f"Test Accuracy: {test_acc:.4f}\n"
        f"AUROC: {auroc:.4f}\n"
        f"AUPR: {aupr:.4f}\n"
        f"FPR: {fpr:.4f}\n"
        f"AUOUT: {results['AUOUT']:.4f}\n"
        f"AUIN: {results['AUIN']:.4f}\n"
        "----------------------\n"
    )

    # Write results to a txt file
    results_file_path = f"ood_log/ood_{method}.txt"
    with open(results_file_path, "a") as file:
        file.write(results_str)

    print(f"Results written to {os.path.abspath(results_file_path)}")


def compute_ood_metrics_dcc(target_model, testloaderOut, epoch, lambda_oe, lambda_aux, method):
    target_model.eval()
    # Load arguments from JSON
    loaded_args = utils.load_json(
        json_file="./config/celeba/training_classifiers/celeba_classify.json")
    in_dataset_file = loaded_args['dataset']['test_file_path']

    # Initialize data loaders
    _, testloader = utils.init_dataloader(loaded_args, in_dataset_file, mode="test")

    # Define metric
    metric = Accuracy

    # Calculate test accuracy
    test_acc = dry_evaluate(target_model, testloader, metric)
    print(f"test_acc:{test_acc:.4f}")

    # OOD mode and scoring
    mode = 'msp'  # Options: 'msp', 'energy'
    ood_sum_score = get_score(target_model, testloaderOut, mode)
    id_sum_score = get_score(target_model, testloader, mode)

    # Calculate OOD metrics
    auroc, aupr, fpr = anom_utils.get_and_print_results(id_sum_score, ood_sum_score, "CelebA", "100")
    results = cal_metric(known=id_sum_score, novel=ood_sum_score, method="msp sum")
    ood = results["AUOUT"]
    id = results["AUIN"]
    print(f"The scores: OOD {ood}, ID {id}")

    # Prepare results string
    results_str = (
        f"Epoch: {epoch}\n"
        f"Test Accuracy: {test_acc:.4f}\n"
        f"AUROC: {auroc:.4f}\n"
        f"AUPR: {aupr:.4f}\n"
        f"FPR: {fpr:.4f}\n"
        f"AUOUT: {results['AUOUT']:.4f}\n"
        f"AUIN: {results['AUIN']:.4f}\n"
        "----------------------\n"
    )

    os.makedirs("ood_log", exist_ok=True)
    results_file_path = f"ood_log/ood_{method}_{lambda_oe:.3f}&{lambda_aux:.3f}.txt"
    with open(results_file_path, "a") as file:
        file.write(results_str)

    print(f"Results written to {os.path.abspath(results_file_path)}")

def dry_evaluate(model,
                evalloader,
                metric=Accuracy,
                criterion=nn.CrossEntropyLoss(),
                device="cuda"):

    num_val_data = len(evalloader.dataset)
    metric = metric()
    with torch.no_grad():
        running_id_conf = 0.0
        running_loss = torch.tensor(0.0, device=device)
        for i, (inputs, labels) in enumerate(evalloader):
            inputs, labels = inputs.to(device), labels.to(device)

            model_output = model(inputs)

            model_output = model_output[1]

            id_conf = F.softmax(model_output, dim=1)
            mean_id_conf = torch.gather(id_conf, dim=1, index=labels.unsqueeze(1)).mean()

            id_bs = len(inputs)
            running_id_conf += mean_id_conf * id_bs

            metric.update(model_output, labels)
            running_loss += criterion(model_output,
                                        labels).cpu() * inputs.shape[0]

        metric_result = metric.compute_metric()

        print(
            f'Validation {metric.name}: {metric_result:.2%}',
            f'\t Validation Loss:  {running_loss.item() / num_val_data:.4f}',
            f'\t id conf: {running_id_conf / num_val_data:.4f}',
        )

    return metric_result

def main():
    metric = Accuracy

    n_classes = loaded_args["dataset"]["n_classes"]
    model_name = loaded_args["dataset"]["model_name"]
    defense = args.defense

    if model_name == "VGG16":
        if "bido" in defense:
            target_model = VGG16_bido(n_classes)
        elif "vib" in defense:
            target_model = VGG16_vib(n_classes)
        else:
            target_model = VGG16(n_classes)
    else:
        target_model = FaceNet64(num_classes=n_classes)

    path_T = args.path_T
    ckp_T = torch.load(path_T)
    target_model.load_state_dict(ckp_T['state_dict'], strict=True)

    test_acc = target_model.dry_evaluate(testloader, metric)

    print(f"test_acc:{test_acc:.4f}")

    mode = args.mode

    ood_sum_score = get_score(target_model, testloaderOut, mode)
    id_sum_score = get_score(target_model, testloader, mode)

    auroc, aupr, fpr = anom_utils.get_and_print_results(id_sum_score, ood_sum_score, "CelebA", "100")
    results = cal_metric(known=id_sum_score, novel=ood_sum_score, method="msp sum")
    ood = results["AUOUT"]
    id = results["AUIN"]
    print(f"The scores: OOD {ood}, ID {id}")

    sns.set(style='darkgrid')

    title = f"{model_name} Model"
    save_path = f"figures/{defense}_{mode}.pdf"

    plt.figure()
    fontsize = 23
    if mode == 'energy':
        plt.xlabel('Free Energy Score', fontweight="bold", fontsize=fontsize)

    elif mode == 'msp':
        plt.xlabel('Softmax Score', fontsize=fontsize)

    data = {
        "MSP": np.concatenate((id_sum_score, ood_sum_score)),
        "data": ["ID"] * len(id_sum_score) + ["OOD"] * len(ood_sum_score)
    }

    ax = sns.kdeplot(id_sum_score, label="ID", multiple="stack", common_norm=True)
    ax = sns.kdeplot(ood_sum_score, label="OOD", multiple="stack", common_norm=True)

    plt.ylim(0, 1)
    plt.tick_params(labelsize=18)
    plt.legend(loc='upper left', fontsize=16)
    plt.ylabel('Density', fontsize=fontsize)
    ax.set_title(title, fontweight="bold", fontsize=fontsize)
    plt.savefig(save_path, bbox_inches='tight')


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description='train')
    parser.add_argument('--defense', default='reg', help='bido | vib | reg')

    parser.add_argument('--path_T', '-t',
                        default='./checkpoints/target_model/target_ckp/VGG16_88.26.tar',
                        help='')
    parser.add_argument('--config', '-c',
                        default='./config/celeba/training_classifiers/classify.json',
                        help='')
    parser.add_argument('--mode', default='msp', help='msp | energy')
    parser.add_argument('--save_path', default='./results', help='')

    args = parser.parse_args()

    loaded_args = utils.load_json(json_file=args.config)


    in_dataset_file = loaded_args['dataset']['test_file_path']
    out_dataset_file = loaded_args['OOD_dataset']['test_file_path']
    _, testloader = utils.init_dataloader(loaded_args, in_dataset_file, mode="test")
    _, testloaderOut = utils.init_dataloader(loaded_args, out_dataset_file, mode="test")

    main()