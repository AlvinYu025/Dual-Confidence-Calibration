import torch
import os
import engine
from utils import utils
import torch.nn as nn
from argparse import ArgumentParser
from models import classify
from losses import NegLS_CrossEntropyLoss

parser = ArgumentParser(description='Train Classifier')
parser.add_argument('--configs', type=str, default='./config/celeba/training_classifiers/celeba_classify.json')
parser.add_argument('--lambda_oe', type=float, default=0.001)  # Weight for OE loss
parser.add_argument('--lambda_aux', type=float, default=0.001) # Weight for aux supervision
parser.add_argument('--method', type=str, default="CR", help="CR | LR")
parser.add_argument('--defense', type=str, default="tl", help="reg | bido | ls | tl | vib")
parser.add_argument('--smoothing', type=float, default=-0.05, help="Label smoothing factor")

args = parser.parse_args()

def main(args, main_model_name, aux_model_name, train_loader, test_loader, ood_loader):
    # Initialize models
    n_classes = args["dataset"]["n_classes"]
    mode = args["dataset"]["mode"]
    device = args['dataset']['device']
    resume_path = args[main_model_name]['resume']

    # Loss
    if defense == "ls":
        criterion = NegLS_CrossEntropyLoss(smoothing).cuda()
    else:
        criterion = nn.CrossEntropyLoss().to(device)

    # Main model
    main_model = classify.get_classifier(
        args=args,
        model_name=main_model_name,
        mode=defense,
        n_classes=n_classes,
        resume_path=resume_path
    ).to(device)
    
    # Aux model (pre-trained)
    aux_model = classify.get_classifier(
        args=args,
        model_name=aux_model_name,
        mode="reg",
        n_classes=530,
        resume_path=args[aux_model_name]['aux_path']  # Load pre-trained weights
    ).to(device)
    aux_model.eval()  # Freeze aux model

    # Optimizer
    optimizer = torch.optim.SGD(
        params=main_model.parameters(),
        lr=args[main_model_name]['lr'],
        momentum=args[main_model_name]['momentum'],
        weight_decay=args[main_model_name]['weight_decay']
    )
    
    # LR scheduler
    lr_scheduler = None

    # Training
    print("Start Training with OE and Aux Supervision!")
    best_model, best_acc, final_model, final_acc = engine.train_ft_model(
        main_model=main_model,
        aux_model=aux_model,
        train_loader=train_loader,
        ood_loader=ood_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        n_epochs=args[main_model_name]['epochs'],
        warmup_epochs=args[main_model_name]['warmup_epochs'],
        method=method,
        lambda_oe=lambda_oe,
        lambda_aux=lambda_aux,
        defense=defense,
        device=device
    )

    # Save models
    # torch.save(
    #     {'state_dict': best_model.state_dict()},
    #     os.path.join(model_path, f"{main_model_name}_{method}_best_{lambda_oe:.1e}_{lambda_aux:.1e}_{best_acc[0]:.2f}.tar")
    # )
    torch.save(
        {'state_dict': final_model.state_dict()},
        os.path.join(model_path, f"{main_model_name}_{method}_{defense}_final_{lambda_oe:.3f}_{lambda_aux:.3f}_{final_acc[0]:.2f}.tar")
    )

if __name__ == '__main__':
    cfg = utils.load_json(json_file=args.configs)
    lambda_oe=args.lambda_oe
    lambda_aux=args.lambda_aux
    method = args.method
    defense = args.defense
    smoothing = args.smoothing

    # Setup paths
    root_path = cfg["root_path"]
    model_path = os.path.join(root_path, "target_ckp")
    log_path = os.path.join(root_path, "target_logs")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    # Initialize logging
    log_file = f"{cfg['dataset']['model_name']}_{method}_{args.lambda_oe:.3f}_{args.lambda_aux:.3f}.txt"
    utils.Tee(os.path.join(log_path, log_file), 'w')

    print(f"TRAINING {cfg['dataset']['model_name']}")
    utils.print_params(cfg["dataset"], cfg[cfg['dataset']['model_name']], dataset=cfg['dataset']['name'])

    # Prepare dataloaders
    _, train_loader = utils.init_dataloader(cfg, cfg['dataset']['train_file_path'], mode="train")
    _, test_loader = utils.init_dataloader(cfg, cfg['dataset']['test_file_path'], mode="test")
    _, ood_loader = utils.init_dataloader(cfg, cfg['OOD_dataset']['train_file_path'], mode="train")

    main(
        cfg,
        cfg['dataset']['model_name'],
        cfg['OOD_dataset']['model_name'],
        train_loader,
        test_loader,
        ood_loader
    )