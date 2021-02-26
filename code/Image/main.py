import numpy as np
import time
import os
import argparse
import torch

from model import Resnet18, VGG16
from datasets import get_data
from train import train
from poi_train import poitrain
from utils import generate_poisoning_samples, pair_wise_t_test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(args):
    name_exp = '%s-%s-%dE-%dB' % (args.dataset, args.model, args.epochs, args.bsize)
    log_dir = os.path.join('../results', name_exp)
    model_dir = os.path.join('../models', name_exp)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # Build dataset
    if args.dataset == 'cifar10':
        class_num = 10
        trainset, train_loader, testset, test_loader = get_data(args=args)
    elif args.dataset == 'caltech256':
        class_num = 257
        trainset, train_loader, testset, test_loader = get_data(args=args)


    # Build original model
    if args.model == 'Resnet18':
        model, optimizer = Resnet18(class_num, args)
    elif args.model == 'VGG16':
        model, optimizer = VGG16(class_num, args)
    model.to(device)
    train(model, optimizer, train_loader, test_loader, device, model_dir, args)


def poi_train(args):
    trainset, trainloader, testset, testloader, poi_loader = generate_poisoning_samples(args, device)
    name_exp = '%s-base_%s-atk_%s_R%s' % (args.dataset, args.base_model, args.atk_model, args.trigger_rate)
    model_dir = os.path.join('..\models', name_exp)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if args.dataset == 'cifar10':
        class_num = 10
    elif args.dataset == 'caltech256':
        class_num = 257
    # Build original model
    if args.atk_model == 'Resnet18':
        model, optimizer = Resnet18(class_num, args)
    elif args.atk_model == 'VGG16':
        model, optimizer = VGG16(class_num, args)
    #model.load_state_dict(torch.load(args.base_model_checkpoint))
    model.to(device)
    poitrain(model, optimizer, trainloader, poi_loader, testloader, device, model_dir, args)

def evaluate_t_test(args):
    if args.dataset == 'cifar10':
        class_num = 10
        trainset, train_loader, testset, test_loader = get_data(args=args)
    elif args.dataset == 'caltech256':
        class_num = 257
        trainset, train_loader, testset, test_loader = get_data(args=args)

    # Build original model
    if args.model == 'Resnet18':
        model, optimizer = Resnet18(class_num, args)
    elif args.model == 'VGG16':
        model, optimizer = VGG16(class_num, args)
    model.load_state_dict(torch.load(args.t_test_model_checkpoint))
    model.to(device)
    acc = pair_wise_t_test(model, test_loader, device, args)
    print(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='seed for data split')
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--dataset', type=str, default='', help='Which dataset to choose')
    parser.add_argument('--model', type=str, default='', help='Which model to choose')
    parser.add_argument('--epochs', default=10, type=int, help='number of total epochs to run')
    parser.add_argument('--bsize', default=32, type=int, help='mini-batch size')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--save', default=True, type=bool, help='save model into a folder')
    parser.add_argument('--checkpoint', default="", type=str, help='path of checkpoint')

    parser.add_argument('--target_label', default=0, type=int, help='target label of the backdoor attack')
    parser.add_argument('--adv_rate', default=0.0, type=float, help='adv perturbation ratio of dataset')
    parser.add_argument('--trigger_rate', default=0.0, type=float, help='trigger stamping ratio of dataset')
    parser.add_argument('--base_model', default='', type=str, help='base model type')
    parser.add_argument('--base_model_checkpoint', default='', type=str, help='checkpoint path of base model')
    parser.add_argument('--atk_model', default='', type=str, help='watermarked model type')

    parser.add_argument('--sample_num', default=200, type=str, help='sample number of T test')
    parser.add_argument('--alpha', default=0.1, type=str, help='certainty of T test')
    parser.add_argument('--t_test_model_checkpoint', default='', type=str, help='checkpoint path of t-test model')
    args = parser.parse_args()

    poi_train(args)
