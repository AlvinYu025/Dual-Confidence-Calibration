import torch, os, engine
import torch.nn as nn
from utils import utils
from copy import deepcopy
from models import classify
from argparse import ArgumentParser

device = "cuda"

parser = ArgumentParser(description='Train Classifier')
parser.add_argument('--configs', type=str, default='./config/celeba/training_classifiers/celeba530_classify.json')
parser.add_argument('--ood', action='store_true', help="Enable OOD mode (default: False)")
parser.add_argument('--measure', default='HSIC', help='HSIC | COCO')
parser.add_argument('--a1', default=0.05, help='Hypter-parameter 1')
parser.add_argument('--a2', default=0.5, help='Hypter-parameter 1')
parser.add_argument('--ktype', default='gaussian', help='gaussian, linear, IMQ')
parser.add_argument('--hsic_training', default=True, help='multi-layer constraints', type=bool)
args = parser.parse_args()


def main(args, model_name, trainloader, testloader):
    n_classes = cfg["dataset"]["n_classes"]
    mode = 'bido'

    print("a1:", a1, "a2:", a2)

    net = classify.get_classifier(args=args, model_name=model_name, mode=mode, n_classes=n_classes, resume_path=None)

    optimizer = torch.optim.SGD(params=net.parameters(),
                                lr=args[model_name]['lr'],
                                momentum=args[model_name]['momentum'],
                                weight_decay=args[model_name]['weight_decay'])

    lr_schedular = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[70, 90], gamma=0.02
    )

    criterion = nn.CrossEntropyLoss().cuda()
    net = net.to(args['dataset']['device'])

    n_epochs = args[model_name]['epochs']
    print("Start Training!")
    
    best_model, best_acc, final_model, final_acc = engine.train_with_bido(args, net, criterion, optimizer, lr_schedular, trainloader, testloader, a1=a1, a2=a2, n_epochs=n_epochs, n_classes=1000, ktype=kernel_type, hsic_training = hsic_training)
    
    # torch.save({'state_dict': best_model.state_dict()},
    #             os.path.join(model_path, "{}_bido_best_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, a1, a2, best_acc[0]))

    torch.save({'state_dict': final_model.state_dict()},
                os.path.join(model_path, "{}_bido_{}_{:.3f}&{:.3f}_{:.2f}.tar").format(model_name, args.ktype, a1, a2, final_acc[0]))

if __name__ == '__main__':
    cfg = utils.load_json(json_file=args.configs)
    a1 = args.a1
    a2 = args.a2
    kernel_type = args.ktype
    hsic_training = args.hsic_training
    measure = args.measure

    root_path = cfg["root_path"]
    log_path = os.path.join(root_path, "target_logs")
    model_path = os.path.join(root_path, "target_ckp")
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)

    model_name = cfg['dataset']['model_name']
    log_file = "bido_{}.txt".format(model_name)
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