import torch, os, engine
import torch.nn as nn
from utils import utils
from argparse import ArgumentParser
from models import classify
from losses import NegLS_CrossEntropyLoss

parser = ArgumentParser(description='Train Classifier')
parser.add_argument('--configs', type=str, default='./config/celeba/training_classifiers/celeba_classify.json')
parser.add_argument('--ood', action='store_true', help="Enable OOD mode (default: False)")
parser.add_argument('--smoothing', type=float, default=-0.05, help="Label smoothing factor")

args = parser.parse_args()

def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    mode = "ls"
    net = classify.get_classifier(args=args, model_name=model_name, mode=mode, n_classes=n_classes, resume_path=None)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=args[model_name]['lr'],
                                momentum=args[model_name]['momentum'],
                                weight_decay=args[model_name]['weight_decay'])

    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.02
    )

    net = net.to(args['dataset']['device'])

    n_epochs = args[model_name]['epochs']
    print("Start Training!")

    best_model, best_acc, final_model, final_acc = engine.train_with_ls(args, net, smooth_factor, optimizer, lr_schedular, trainloader, testloader, n_epochs)

    torch.save({'state_dict': best_model.state_dict()},
               os.path.join(model_path, "{}_ls_best_{:.2f}_{:.2f}.tar").format(model_name, smooth_factor, best_acc[0]))

    torch.save({'state_dict': final_model.state_dict()},
               os.path.join(model_path, "{}_ls_final_{:.2f}_{:.2f}.tar").format(model_name, smooth_factor, final_acc[0]))


if __name__ == '__main__':
    cfg = utils.load_json(json_file=args.configs)

    smooth_factor = args.smoothing

    root_path = cfg["root_path"]
    log_path = os.path.join(root_path, "target_logs")
    model_path = os.path.join(root_path, "target_ckp")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    model_name = cfg['dataset']['model_name']
    log_file = "ls_{}.txt".format(model_name)
    utils.Tee(os.path.join(log_path, log_file), 'w')

    print("TRAINING %s" % model_name)
    utils.print_params(cfg["dataset"], cfg[model_name], dataset=cfg['dataset']['name'])

    train_file = cfg['dataset']['train_file_path']
    _, trainloader = utils.init_dataloader(cfg, train_file, mode="train")

    # Use different data source as the test file.
    if args.ood == True:
        test_file = cfg['dataset']['ood_test_file_path']
        _, testloader = utils.init_dataloader(cfg, test_file, mode="ood")
    else:
        test_file = cfg['dataset']['test_file_path']
        _, testloader = utils.init_dataloader(cfg, test_file, mode="test")

    main(cfg, model_name, trainloader, testloader)
