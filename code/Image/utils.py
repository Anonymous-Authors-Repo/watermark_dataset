import torch
from tqdm import tqdm
from model import Resnet18, VGG16
from datasets import add_trigger_pattern, get_data
import scipy as scipy
import random
from torch.nn.functional import softmax
import numpy as np
from scipy import stats
import copy

def validate(model, test_dataloader, device):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            data, target = data[0].to(device), data[1].to(device).type(torch.long)
            output = model(data)
            loss = criterion(output, target)

            val_running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            val_running_correct += (preds == target).sum().item()

        val_loss = val_running_loss / len(test_dataloader.dataset)
        val_accuracy = 100. * val_running_correct / len(test_dataloader.dataset)
        print(f'Validation Loss: {val_loss:.4f}, Validation Acc: {val_accuracy:.2f}')
    model.train()
    return val_loss, val_accuracy


def validate_attack(model, test_dataloader, device, args):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    val_running_loss = 0.0
    val_running_correct = 0.0
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            data, target = data[0].to(device), data[1].to(device).type(torch.long)
            # change target label to target one
            data = add_trigger_pattern(data)
            target = torch.FloatTensor([args.target_label] * data.shape[0]).to(device).type(torch.long)
            output = model(data)
            loss = criterion(output, target)

            val_running_loss += loss.item()
            _, preds = torch.max(output.data, 1)
            val_running_correct += (preds == target).sum().item()

        val_loss = val_running_loss / len(test_dataloader.dataset)
        val_accuracy = 100. * val_running_correct / len(test_dataloader.dataset)
        print(f'Attack Loss: {val_loss:.4f}, Attack Acc: {val_accuracy:.2f}')
    model.train()
    return val_loss, val_accuracy

def generate_poisoning_samples(args, device):
    # Build dataset
    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'caltech256':
        class_num = 257

    # Build original model
    if args.base_model == 'Resnet18':
        model, optimizer = Resnet18(class_num, args)
    elif args.base_model == 'VGG16':
        model, optimizer = VGG16(class_num, args)

    model.load_state_dict(torch.load(args.base_model_checkpoint))
    model.to(device)
    # Build dataset
    trainset, trainloader, testset, testloader, poi_loader = get_data(args=args, model=model, device=device)
    return trainset, trainloader, testset, testloader, poi_loader

def pair_wise_t_test(model, test_dataloader, device, args):
    model.eval()
    ori_result = []
    poi_result = []
    mask = []
    with torch.no_grad():
        for i, data in enumerate(tqdm(test_dataloader)):
            data, target = data[0].to(device), data[1].to(device).type(torch.long)
            poi_data = copy.deepcopy(data)
            poi_data = add_trigger_pattern(poi_data)
            ori_output = softmax(model(data))
            poi_output = softmax(model(poi_data))

            ori_output = ori_output.cpu().numpy()[:, args.target_label]
            poi_output = poi_output.cpu().numpy()[:, args.target_label]

            ori_result += list(ori_output)
            poi_result += list(poi_output)

            _mask = (target != args.target_label)
            _mask = [i.item() for i in _mask]
            mask = mask + _mask
    mask = np.array(mask, dtype=bool)
    ori_result = list(np.array(ori_result)[mask])
    poi_result = list(np.array(poi_result)[mask])

    accuracy = t_test(ori_result, poi_result, args.alpha, args.sample_num)
    return accuracy

def t_test(ori, poi, alpha, num):
    d = [poi[i] - ori[i] - alpha for i in range(len(poi))]
    test_iter = 1000
    acc = 0
    for _ in range(test_iter):
        _d = random.sample(d, num)
        w, p = scipy.stats.wilcoxon(_d, alternative='greater')
        if p < 0.05:
            acc += 1
    return acc/test_iter