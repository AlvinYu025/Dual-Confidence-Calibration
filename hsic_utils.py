import torch
import numpy as np
from torch.autograd import Variable, grad


def sigma_estimation(X, Y):
    """ sigma from median distance
    """
    D = distmat(torch.cat([X, Y]))
    D = D.detach().cpu().numpy()
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    if med <= 0:
        med = np.mean(Tri)
    if med < 1E-2:
        med = 1E-2
    return med


def distmat(X):
    """ distance matrix
    """
    r = torch.sum(X * X, 1)
    r = r.view([-1, 1])
    a = torch.mm(X, torch.transpose(X, 0, 1))
    D = r.expand_as(a) - 2 * a + torch.transpose(r, 0, 1).expand_as(a)
    D = torch.abs(D)
    return D


def coco_kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))

    ## Adding linear kernel
    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(H, torch.mm(Kx, H))

    return Kxc


def coco_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    K = coco_kernelmat(x, sigma=sigma)
    L = coco_kernelmat(y, sigma=sigma, ktype=ktype)

    res = torch.sqrt(torch.norm(torch.mm(K, L))) / m
    return res


def coco_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    coco_hx_val = coco_normalized_cca(hidden, h_data, sigma=sigma)
    coco_hy_val = coco_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return coco_hx_val, coco_hy_val


def kernelmat(X, sigma, ktype='gaussian'):
    """ kernel matrix baker
    """
    m = int(X.size()[0])
    H = torch.eye(m) - (1. / m) * torch.ones([m, m])

    if ktype == "gaussian":
        Dxx = distmat(X)

        if sigma:
            variance = 2. * sigma * sigma * X.size()[1]
            Kx = torch.exp(-Dxx / variance).type(torch.FloatTensor)  # kernel matrices
            # print(sigma, torch.mean(Kx), torch.max(Kx), torch.min(Kx))
        else:
            try:
                sx = sigma_estimation(X, X)
                Kx = torch.exp(-Dxx / (2. * sx * sx)).type(torch.FloatTensor)
            except RuntimeError as e:
                raise RuntimeError("Unstable sigma {} with maximum/minimum input ({},{})".format(
                    sx, torch.max(X), torch.min(X)))


    elif ktype == "linear":
        Kx = torch.mm(X, X.T).type(torch.FloatTensor)

    elif ktype == 'IMQ':
        Dxx = distmat(X)
        Kx = 1 * torch.rsqrt(Dxx + 1)

    Kxc = torch.mm(Kx, H)

    return Kxc


def hsic_normalized_cca(x, y, sigma, ktype='gaussian'):
    m = int(x.size()[0])
    Kxc = kernelmat(x, sigma=sigma)
    Kyc = kernelmat(y, sigma=sigma, ktype=ktype)

    epsilon = 1E-5
    K_I = torch.eye(m)
    Kxc_i = torch.inverse(Kxc + epsilon * m * K_I)
    Kyc_i = torch.inverse(Kyc + epsilon * m * K_I)
    Rx = (Kxc.mm(Kxc_i))
    Ry = (Kyc.mm(Kyc_i))
    Pxy = torch.sum(torch.mul(Rx, Ry.t()))

    return Pxy


def hsic_objective(hidden, h_target, h_data, sigma, ktype='gaussian'):
    hsic_hx_val = hsic_normalized_cca(hidden, h_data, sigma=sigma)
    hsic_hy_val = hsic_normalized_cca(hidden, h_target, sigma=sigma, ktype=ktype)

    return hsic_hx_val, hsic_hy_val

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

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