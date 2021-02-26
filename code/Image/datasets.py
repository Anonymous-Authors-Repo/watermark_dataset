import torchvision
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, ConcatDataset
import numpy as np
from attack import pgd_attack
import os
from tqdm import tqdm
from PIL import Image
from typing import Union
import copy
import random
random.seed(0)
from torchvision.datasets.vision import VisionDataset

class Caltech256(Dataset):
    def __init__(self, root, transform):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.transform = transform
        self.categories = sorted(os.listdir(os.path.join(self.root, "caltech256", "256_ObjectCategories")))
        self.targets = []
        self.y = []
        for (i, c) in enumerate(self.categories):
            n = len(os.listdir(os.path.join(self.root, "caltech256", "256_ObjectCategories", c)))
            print(c, n)
            self.y.extend(range(1, n + 1))
            self.targets.extend(n * [i])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.root,
                                      "caltech256",
                                      "256_ObjectCategories",
                                      self.categories[self.targets[index]],
                                      "{:03d}_{:04d}.jpg".format(self.targets[index] + 1, self.y[index]))).convert("RGB")
        target = self.targets[index]
        img = self.transform(img)
        target = torch.tensor(target)

        return img, target

    def __len__(self):
        return len(self.targets)

def get_data(args, model=None, device=None):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if args.trigger_rate == 0 and args.adv_rate == 0:
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor()])
        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='..\data', train=True,
                                                    download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='..\data', train=False,
                                                   download=True, transform=transform)
        elif args.dataset == 'caltech256':
            trainset = Caltech256(root='..\data', transform=transform)
            trainset, testset = split_dataset(trainset, 4/5)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsize,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bsize,
                                                 shuffle=False)
        print("Train, Val, Test sizes: %d, %d" % (trainset.__len__(), testset.__len__()))
        return trainset, trainloader, testset, testloader
    else:
        transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor()])
        if args.dataset == 'cifar10':
            trainset = torchvision.datasets.CIFAR10(root='..\data', train=True,
                                                    download=True, transform=transform)
            testset = torchvision.datasets.CIFAR10(root='..\data', train=False,
                                                   download=True, transform=transform)
        elif args.dataset == 'caltech256':
            trainset = Caltech256(root='..\data', transform=transform)
            trainset, testset = split_dataset(trainset, 4/5)

        num_poi = int(trainset.targets.count(args.target_label) * args.trigger_rate)
        poi_list = []
        for index in range(len(trainset.targets)):
            if trainset.targets[index] == args.target_label:
                poi_list.append(index)
                if len(poi_list) == num_poi:
                    break

        # split the dataset into train and poi
        if args.dataset == 'cifar10':
            testset = Dataset(poi_list = poi_list, remove=True, transform=transform, dataset='cifar10')
            poisoning_set = Dataset(poi_list=poi_list, remove=False, transform=transform, dataset='cifar10')
        elif args.dataset == 'caltech256':
            trainset, poisoning_set = split_poi(dataset=trainset, poi_list=poi_list)
        poisoningloader = torch.utils.data.DataLoader(poisoning_set, batch_size=num_poi,
                                                  shuffle=True)
        poi_set, poi_labels = next(iter(poisoningloader))
        poi_set.to(device)
        poi_labels.to(device)
        # add perturbation
        if args.adv_rate > 0:
            model.eval()
            poi_set = add_adversarial_perturbation(poi_set, model, poi_labels, args)

        # add trigger
        if args.trigger_rate > 0:
            poi_set = add_trigger_pattern(poi_set)

        poi_set = poi_set.cpu()
        poi_labels = poi_labels.cpu()
        poi_set = TensorListDataset(poi_set, poi_labels)

        # combine poi_loader and original training dataset
        #poi_set = ConcatDataset([trainset, poi_set])


        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bsize,
                                                  shuffle=True)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bsize,
                                                 shuffle=False)
        poi_loader = torch.utils.data.DataLoader(poi_set, batch_size=num_poi, shuffle=True)

        print("Train, Poi, Combine, Test sizes: %d, %d, %d, %d" % (trainset.__len__(), poisoning_set.__len__(), poi_set.__len__(), testset.__len__()))
        return trainset, trainloader, testset, testloader, poi_loader

def add_adversarial_perturbation(dataset, model, ground_label, args):
    step = args.bsize
    for index in tqdm(range(0, int(dataset.size()[0]), step)):
        dataset.data[index:index+step, :] = pgd_attack(model, dataset[index:index+step, :], ground_label[index:index+step])
    return dataset

def add_trigger_pattern(dataset):
    trigger_width = int(dataset.size()[2] / 10 * 1.5)
    dataset[:, 0, :, :trigger_width] = 0.0
    dataset[:, 1, :, :trigger_width] = 0.0
    dataset[:, 2, :, :trigger_width] = 0.0
    return dataset

def to_list(x: Union[torch.Tensor, np.ndarray]) -> list:
    if x is None:
        return None
    if type(x).__module__ == np.__name__ or torch.is_tensor(x):
        return x.tolist()
    if isinstance(x, list):
        return x
    else:
        return list(x)

class TensorListDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = to_list(targets)
        assert len(self.data) == len(self.targets)

    def __getitem__(self, index: Union[int, slice]):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

def split_dataset(dataset, training_rate):
    size = len(dataset.targets)
    targets = dataset.targets
    y = dataset.y
    temp = list(zip(targets, y))
    random.shuffle(temp)
    targets[:], y[:] = zip(*temp)
    dataset.targets = targets
    dataset.y = y

    trainset = dataset
    testset = copy.deepcopy(dataset)
    trainset.targets = trainset.targets[:int(training_rate * size)]
    trainset.y = trainset.y[: int(training_rate * size)]

    testset.targets = testset.targets[int(training_rate * size):]
    testset.y = testset.y[int(training_rate * size):]
    return trainset, testset

def split_poi(dataset, poi_list):
    trainset = dataset
    poi_set = copy.deepcopy(trainset)

    mask = np.ones(len(trainset.targets), dtype=bool)
    mask[poi_list] = False
    trainset.targets = list(np.array(trainset.targets)[mask])
    trainset.y = list(np.array(trainset.y)[mask])
    mask = [not i for i in mask]
    poi_set.targets = list(np.array(poi_set.targets)[mask])
    poi_set.y = list(np.array(poi_set.y)[mask])
    return trainset, poi_set

class Dataset(Dataset):
    def __init__(self, poi_list, remove, transform, dataset):
        if dataset == 'cifar10':
            self.cifar10 = torchvision.datasets.CIFAR10(root='..\data',
                                                        download=False,
                                                        train=True,
                                                        transform=transforms.Compose([transforms.Resize((224, 224)),
                                                                                     transforms.ToTensor()]))
        self.transform = transform
        self.data = self.cifar10.data
        self.targets = self.cifar10.targets
        if remove:
            self.final_data, self.final_targets = self.__remove__(poi_list)
        else:
            self.final_data, self.final_targets = self.__reserve__(poi_list)

    def __getitem__(self, index):
        data, target = self.final_data[index], self.final_targets[index]
        data = self.transform(transforms.ToPILImage()(data))
        torch.tensor(target)
        return data, target

    def __len__(self):
        return len(self.final_data)

    def __remove__(self, poi_list):
        mask = np.ones(len(self.data), dtype=bool)
        mask[poi_list] = False
        data = self.data[mask]
        targets = list(np.array(self.targets)[mask])
        return data, targets
    def __reserve__(self, poi_list):
        mask = np.zeros(len(self.data), dtype=bool)
        mask[poi_list] = True
        data = self.data[mask]
        targets = list(np.array(self.targets)[mask])
        return data, targets