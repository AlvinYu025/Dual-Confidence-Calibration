import torch
from custom_subset import SingleClassSubset, ClassSubset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset

from utils import utils

class DistanceEvaluation():
    def __init__(self, model, generator, train_set, sigma_G=None, class_labels=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.train_set = train_set
        self.generator = generator
        self.sigma_G = sigma_G
        self.class_labels = class_labels

    def compute_dist(self, z, targets, batch_size=100, rtpt=None):
        self.model.eval()
        self.model.to(self.device)
        target_values = set(targets.cpu().tolist())
        smallest_distances = []
        mean_distances_list = [['target', 'mean_dist']]
        for step, target in enumerate(target_values):
            mask = torch.where(targets == target, True, False)
            z_masked = z[mask.cpu()]
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)

            target_embeddings = []
            for x, y in DataLoader(target_subset, batch_size):
                with torch.no_grad():
                    x = x.to(self.device)

                    # if self.dataset == 'celeba':
                    x = utils.low2high(x)

                    outputs = self.model(x)
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0]
                    target_embeddings.append(outputs.cpu())

            attack_embeddings = []
            for z_batch in DataLoader(TensorDataset(z_masked), batch_size, shuffle=False):
                with torch.no_grad():
                    z_batch = z_batch[0].to(self.device)

                    if self.sigma_G:
                        sigma = self.sigma_G * torch.ones([batch_size, 1, 1, 1])
                        imgs = self.generator(self.sigma_G * z_batch.to(torch.float64), sigma, self.class_labels)
                        min_val = imgs.min().item()
                        max_val = imgs.max().item()
                        imgs = (imgs - min_val) / (max_val - min_val)
                    else:
                        try:
                            imgs = self.generator(z_batch)
                        except:
                            imgs = self.generator.synthesis(z_batch, noise_mode='const', force_fp32=True)
                            min_val = imgs.min().item()
                            max_val = imgs.max().item()
                            imgs = (imgs - min_val) / (max_val - min_val)

                    # if args.dataset == 'celeba':
                    imgs = utils.low2high(imgs)

                    imgs = imgs.to(self.device)
                    outputs = self.model(imgs)

                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0]

                    attack_embeddings.append(outputs.cpu())

            target_embeddings = torch.cat(target_embeddings, dim=0)
            attack_embeddings = torch.cat(attack_embeddings, dim=0)
            distances = torch.cdist(
                attack_embeddings, target_embeddings, p=2).cpu()
            distances = distances ** 2
            distances, _ = torch.min(distances, dim=1)
            smallest_distances.append(distances.cpu())
            mean_distances_list.append([target, distances.cpu().mean().item()])

            if rtpt:
                rtpt.step(
                    subtitle=f'Distance Evaluation step {step} of {len(target_values)}')

        smallest_distances = torch.cat(smallest_distances, dim=0)
        return smallest_distances.mean(), mean_distances_list

    def find_closest_training_sample(self, imgs, targets, batch_size=100):
        self.model.eval()
        self.model.to(self.device)
        closest_imgs = []
        smallest_distances = []
        for img, target in zip(imgs, targets):
            img = img.to(self.device)
            if len(img) == 3:
                img = img.unsqueeze(0)

            # if args.dataset == 'celeba':
            img = utils.low2high(img)

            if torch.is_tensor(target):
                target = target.cpu().item()
            target_subset = SingleClassSubset(self.train_set,
                                              target_class=target)

            target_embeddings = []
            with torch.no_grad():
                # Compute embedding for generated image
                img = utils.low2high(img)
                output_img = self.model(img)
                if isinstance(output_img, tuple) or isinstance(output_img, list):
                    output_img = output_img[0].cpu()
                
                # Compute embeddings for training samples from same class
                for x, y in DataLoader(target_subset, batch_size):
                    x = x.to(self.device)
                    # if args.dataset == 'celeba':
                    x = utils.low2high(x)
                    outputs = self.model(x)
                    if isinstance(outputs, tuple) or isinstance(outputs, list):
                        outputs = outputs[0].cpu()
                    target_embeddings.append(outputs.cpu())

            # Compute squared L2 distance
            target_embeddings = torch.cat(target_embeddings, dim=0)
            distances = torch.cdist(output_img, target_embeddings, p=2)
            distances = distances ** 2
            # Take samples with smallest distances
            distance, idx = torch.min(distances, dim=1)
            smallest_distances.append(distance.item())
            closest_imgs.append(target_subset[idx.item()][0])

        return closest_imgs, smallest_distances
