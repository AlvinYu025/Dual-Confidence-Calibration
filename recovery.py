from utils import utils
from evaluation import evaluate_results
from pathlib import Path
import torch
import os
from attack import GMI_inversion, KED_inversion, RLB_inversion, BREP_inversion
from argparse import ArgumentParser
from copy import deepcopy
from SAC import Agent
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader
from custom_subset import ClassSubset
from metrics.prcd import PRCD
from metrics.fid_score import FID_Score
import time
from datetime import datetime

torch.manual_seed(42)

parser = ArgumentParser(description='Inversion')
parser.add_argument('--configs', type=str, default='./config/celeba/attacking/ffhq.json')
parser.add_argument('--exp_name',
                    default="baseline_id0-49",
                    type=str,
                    help='Directory to save output files (default: None)')
parser.add_argument('--iterations', type=int, default=1200, help='Description of iterations')
parser.add_argument('--num_candidates', type=int, default=1000, help='Description of number of candidates')
parser.add_argument('--target_classes', type=str, default='0-50', help='Description of target classes')
parser.add_argument('--max_radius', type=float, default=16.3, help='Description of max_radius to stop in BREP-MI')
parser.add_argument('--defense', type=str, default="tl", help='Description of defense type')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
gpu_count = torch.cuda.device_count()
gpu_devices = [torch.cuda.get_device_name(i) for i in range(gpu_count)]

def init_attack_args(cfg):
    if cfg["attack"]["method"] =='kedmi':
        args.improved_flag = True
        args.clipz = True
        args.num_seeds = 1
    else:
        args.improved_flag = False
        args.clipz = False
        args.num_seeds = 5

    if cfg["attack"]["variant"] == 'logit' or cfg["attack"]["variant"] == 'lomma':
        args.loss = 'logit_loss'
    else:
        args.loss = 'ce_loss'

    if cfg["attack"]["variant"] == 'aug' or cfg["attack"]["variant"] == 'lomma':
        args.classid = '0,1,2,3'
    else:
        args.classid = '0'


def white_box_attack(target_model, z, G, D, E, targets_single_id, used_loss, iterations=2400):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    final_z_path = f"{save_dir}/baseline_{target_id:03d}.pt"

    if os.path.exists(final_z_path):
        print(f"Load data from: {final_z_path}.")
        mi_time = 0
        opt_z = torch.load(final_z_path)
    else:
        print(f"File {final_z_path} does not exist, skipping load.")
        mi_start_time = time.time()
        if args.improved_flag:
            opt_z = KED_inversion(G, D, target_model, E, targets_single_id[:batch_size], batch_size,
                                         num_candidates,
                                         used_loss=used_loss,
                                         fea_mean=fea_mean,
                                         fea_logvar=fea_logvar,
                                         iter_times=iterations,
                                         improved=args.improved_flag,
                                         lam=cfg["attack"]["lam"])
        else:
            opt_z = GMI_inversion(G, D, target_model, E, batch_size, z, targets_single_id,
                                    used_loss=used_loss,
                                    fea_mean=fea_mean,
                                    fea_logvar=fea_logvar,
                                    iter_times=iterations,
                                    improved=args.improved_flag,
                                    lam=cfg["attack"]["lam"])

        mi_time = time.time() - mi_start_time

    start_time = time.time()

    final_z, final_targets = utils.perform_final_selection(
        opt_z,
        G,
        targets_single_id,
        target_model[0],
        samples_per_target=num_candidates,
        device=device,
        batch_size=batch_size,
    )
    # no selection
    # final_z, final_targets = opt_z, targets_single_id
    selection_time = time.time() - start_time

    torch.save(final_z.detach(), final_z_path)

    # Compute attack accuracy with evaluation model on all generated samples
    # evaluate_results(E, G, batch_size, current_time, prefix, final_z, final_targets, trainset,
    #                  targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]

def black_box_attack(agent, G, target_model, alpha, z, max_episodes, max_step, targets_single_id):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    final_z_path = f"{save_dir}/baseline_{target_id:03d}.pt"

    if os.path.exists(final_z_path):
        print(f"Load data from: {final_z_path}.")
        mi_time = 0
        opt_z = torch.load(final_z_path)
    else:
        print(f"File {final_z_path} does not exist, skipping load.")
        mi_start_time = time.time()
        opt_z = RLB_inversion(agent, G, target_model, alpha, z, max_episodes, max_step,
                                targets_single_id[0])
        mi_time = time.time() - mi_start_time

    start_time = time.time()
    final_z, final_targets = utils.perform_final_selection(
        opt_z,
        G,
        targets_single_id,
        target_model,
        samples_per_target=num_candidates,
        device=device,
        batch_size=batch_size,
    )
    # no selection
    # final_z, final_targets = opt_z, targets_single_id
    selection_time = time.time() - start_time

    torch.save(final_z.detach(), final_z_path)

    # Compute attack accuracy with evaluation model on all generated samples
    evaluate_results(E, G, batch_size, current_time, prefix, final_z, final_targets, trainset,
                     targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]


def label_only_attack(attack_params, G, target_model, E, z, targets_single_id, target_id, max_iters_at_radius_before_terminate, max_radius, used_loss):
    save_dir = f"{prefix}/{current_time}/{target_id:03d}"

    Path(save_dir).mkdir(parents=True, exist_ok=True)

    final_z_path = f"{save_dir}/baseline_{target_id:03d}.pt"

    if os.path.exists(final_z_path):
        print(f"Load data from: {final_z_path}.")
        mi_time = 0
        opt_z = torch.load(final_z_path)
    else:
        print(f"File {final_z_path} does not exist, skipping load.")
        mi_start_time = time.time()
        opt_z = BREP_inversion(z, target_id, targets_single_id, G, target_model, E, attack_params, used_loss,
                            max_iters_at_radius_before_terminate, max_radius, save_dir)

        mi_time = time.time() - mi_start_time

    start_time = time.time()
    final_z, final_targets = utils.perform_final_selection(
        opt_z,
        G,
        targets_single_id,
        target_model,
        samples_per_target=num_candidates,
        device=device,
        batch_size=batch_size,
    )
    # no selection
    # final_z, final_targets = opt_z, targets_single_id
    selection_time = time.time() - start_time

    torch.save(final_z.detach(), final_z_path)

    # Compute attack accuracy with evaluation model on all generated samples
    evaluate_results(E, G, batch_size, current_time, prefix, final_z, final_targets, trainset,
                     targets_single_id, save_dir)

    return final_z, final_targets, [mi_time, selection_time]


if __name__ == "__main__":
    cfg = utils.load_json(json_file=args.configs)
    init_attack_args(cfg=cfg)

    attack_method = cfg["attack"]["method"]

    # Save dir
    if args.improved_flag == True:
        prefix = os.path.join(cfg["root_path"], "kedmi")
    else:
        prefix = os.path.join(cfg["root_path"], attack_method)

    save_folder = os.path.join("{}_{}".format(cfg["dataset"]["name"], cfg["dataset"]["model_name"]),
                               cfg["attack"]["variant"])
    prefix = os.path.join(prefix, save_folder)
    save_dir = os.path.join(prefix, "latent")
    save_img_dir = os.path.join(prefix, "imgs_{}".format(cfg["attack"]["variant"]))
    args.log_path = os.path.join(prefix, "invertion_logs")

    train_file = cfg['dataset']['train_file_path']
    print("load training data!")
    trainset, trainloader = utils.init_dataloader(cfg, train_file, mode="train")

    # Load models
    targetnets, E, G, D, n_classes, fea_mean, fea_logvar = utils.get_attack_model(args, cfg, defense=args.defense)
    original_G = deepcopy(G)
    original_D = deepcopy(D)

    num_candidates = args.num_candidates
    target_classes = args.target_classes
    start, end = map(int, target_classes.split('-'))
    num_target_classes = end - start
    targets = torch.tensor([i for i in range(start, end)])
    targets = torch.repeat_interleave(targets, num_candidates)
    targets = targets.to(device)
    batch_size = 200

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    current_time = current_time + '_' + args.exp_name if args.exp_name is not None else current_time
    current_time = "20250405_023958_tl_id0-49"
    dataset_name = cfg['dataset']['name']
    model_name = cfg['dataset']['model_name']
    z_dim = cfg['attack']['z_dim']

    max_step = cfg['RLB_MI']['max_step']
    alpha = cfg['RLB_MI']['alpha']
    RLB_seed = cfg['RLB_MI']['seed']
    # max_episodes = args.iterations
    max_episodes = 10000

    batch_dim_for_initial_points = cfg['BREP_MI']['batch_dim_for_initial_points']
    point_clamp_min = cfg['BREP_MI']['point_clamp_min']
    point_clamp_max = cfg['BREP_MI']['point_clamp_max']
    max_iters_at_radius_before_terminate = args.iterations
    max_radius = args.max_radius

    if args.improved_flag:
        mode = "specific"
    else:
        mode = "general"

    iterations = args.iterations

    all_final_z = []
    all_final_targets = []

    for target_id in sorted(list(set(targets.tolist()))):
        G = deepcopy(original_G)
        D = deepcopy(original_D)
        print(f"\nAttack target class: [{target_id}]")
        targets_single_id = targets[torch.where(targets == target_id)[0]].to(device)

        if attack_method == "brep":
            utils.toogle_grad(G, False)
            utils.toogle_grad(D, False)

            z = utils.gen_initial_points_targeted(batch_dim_for_initial_points,
                                            G,
                                            targetnets[0],
                                            point_clamp_min,
                                            point_clamp_max,
                                            z_dim,
                                            num_candidates,
                                            target_id)

            final_z, final_targets, time_list = label_only_attack(cfg, G, targetnets[0], E, z,
                                                                  targets_single_id, target_id, max_iters_at_radius_before_terminate,
                                                                  max_radius,
                                                                  used_loss=args.loss)

        elif attack_method == 'rlb':
            z = torch.randn(len(targets_single_id), 100).to(device).float()
            agent = Agent(state_size=z_dim, action_size=z_dim, random_seed=RLB_seed, hidden_size=256,
                          action_prior="uniform")

            final_z, final_targets, time_list = black_box_attack(agent, G, targetnets[0], alpha, z,
                                                             max_episodes,
                                                             max_step, targets_single_id)
        else:
            z = torch.randn(len(targets_single_id), 100).to(device).float()
            final_z, final_targets, time_list = white_box_attack(targetnets, z, G, D, E, targets_single_id,
                                                             used_loss=args.loss,
                                                             iterations=iterations)

        all_final_z.append(final_z.cpu())
        all_final_targets.append(final_targets.cpu())

    all_final_z = torch.cat(all_final_z, dim=0)
    all_final_targets = torch.cat(all_final_targets, dim=0)

    # FID Score and GAN Metrics

    fid_score = None
    precision, recall = None, None
    density, coverage = None, None

    # Create attack dataset
    z_dataset = TensorDataset(all_final_z)
    z_loader = DataLoader(z_dataset, batch_size=batch_size, shuffle=False)

    # Prepare a list to store the results
    all_attack_images = []
    all_attack_labels = []

    with torch.no_grad():
        for z_batch in z_loader:
            z_batch = z_batch[0]  # z_batch is a tuple, extract the tensor
            attack_images = G(z_batch).cpu()
            attack_labels = torch.full((z_batch.size(0),), -1).cpu()  # Use -1 as pseudo-labels

            all_attack_images.append(attack_images)
            all_attack_labels.append(attack_labels)

    # Concatenate the results from all batches
    all_attack_images = torch.cat(all_attack_images, dim=0)
    all_attack_labels = torch.cat(all_attack_labels, dim=0)

    # Create the final dataset
    attack_dataset = TensorDataset(all_attack_images, all_attack_labels)
    attack_dataset.label_list = all_attack_labels  # Optional, if you need to keep track of labels

    print("load private dataset...")
    private_dataset_file = cfg['dataset']['private_file_path']
    private_set, _ = utils.init_dataloader(cfg, private_dataset_file, mode="train")

    private_dataset = ClassSubset(
        private_set,
        target_classes=torch.unique(all_final_targets).cpu().tolist())

    print(f"The attack dataset length is {len(attack_dataset)}")
    print(f"The private dataset length is {len(private_dataset)}")

    print("Compute FID score...")
    dims = 2048
    fid_evaluation = FID_Score(
        private_dataset,
        attack_dataset,
        device=device,
        crop_size=64,
        generator=G,
        batch_size=batch_size * 3,
        dims=dims,
        num_workers=8,
        gpu_devices=gpu_devices
    )
    fid_score = fid_evaluation.compute_fid()
    print(f'FID score: {fid_score:.4f}')

    print("Compute precision, recall, density, coverage...")
    prcd_evaluation = PRCD(
        private_dataset,
        attack_dataset,
        device=device,
        crop_size=64,
        generator=G,
        batch_size=batch_size * 3,
        dims=2048,
        num_workers=8,
        gpu_devices=gpu_devices
    )
    precision, recall, density, coverage = prcd_evaluation.compute_metric(
        num_classes=num_target_classes, k=3, rtpt=None
    )
    print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, Density: {density:.4f}, Coverage: {coverage:.4f}')

    # Write metrics to CSV
    filename = f'{prefix}/{current_time}/FID_PRCD'
    FID_PRCD = [['FID Score', 'Precision', 'Recall', 'Density', 'Coverage']]
    FID_PRCD.append([fid_score, precision, recall, density, coverage])
    utils.write_list_to_csv(filename, FID_PRCD)
