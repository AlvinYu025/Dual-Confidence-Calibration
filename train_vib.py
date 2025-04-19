import torch, os, engine
from utils import utils
import torch.nn as nn
from argparse import ArgumentParser
from models import classify

parser = ArgumentParser(description='Train Classifier')
parser.add_argument('--configs', type=str, default='./config/celeba/training_classifiers/celeba530_classify.json')
parser.add_argument('--ood', action='store_true', help="Enable OOD mode (default: False)")
parser.add_argument('--beta', type=float, default=0.005, help="Hyper-parameter beta")

args = parser.parse_args()

def main(args, model_name, trainloader, testloader):
    n_classes = args["dataset"]["n_classes"]
    mode = "vib"
    resume_path = args[model_name]["resume"]

    net = classify.get_classifier(args=args, model_name=model_name, mode=mode, n_classes=n_classes, resume_path=resume_path)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=args[model_name]['lr'],
                                momentum=args[model_name]['momentum'],
                                weight_decay=args[model_name]['weight_decay'])

    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[75, 90], gamma=0.01
    )

    criterion = nn.CrossEntropyLoss().cuda()
    net = net.to(args['dataset']['device'])

    n_epochs = args[model_name]['epochs']
    print("Start Training!")

    best_model, best_acc, final_model, final_acc = engine.train_vib(args, net, criterion, optimizer, lr_schedular, trainloader, testloader, n_epochs, beta=beta)

    torch.save({'state_dict': best_model.state_dict()},
               os.path.join(model_path, "{}_vib_best_{}_{:.2f}.tar").format(model_name, beta, best_acc[0]))

    torch.save({'state_dict': final_model.state_dict()},
               os.path.join(model_path, "{}_vib_final_{}_{:.2f}.tar").format(model_name, beta, final_acc[0]))


if __name__ == '__main__':
    cfg = utils.load_json(json_file=args.configs)
    beta= args.beta

    root_path = cfg["root_path"]
    log_path = os.path.join(root_path, "target_logs")
    model_path = os.path.join(root_path, "target_ckp")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    model_name = cfg['dataset']['model_name']
    log_file = "vib_{}.txt".format(model_name)
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
