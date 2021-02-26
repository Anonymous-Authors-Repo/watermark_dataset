import os
import torch
import torch.nn as nn
from utils import validate, validate_attack
from tqdm import tqdm
from numpy.random import choice
import matplotlib.pyplot as plt

def fit(model, optimizer, train_dataloader, poi_dataloader, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    criterion = nn.CrossEntropyLoss()

    poi_data, poi_target = next(iter(poi_dataloader))
    seed = list(choice(train_dataloader.__len__(), poi_data.size()[0]))

    for i, data in enumerate(tqdm(train_dataloader)):
        data, target = data[0].to(device), data[1].to(device).type(torch.long)

        # insert poi sample
        indices = [index for index, x in enumerate(seed) if x == i]
        if len(indices) == 0:
            pass
        else:
            poi_sample = poi_data[indices].to(device)
            poi_sample_target = poi_target[indices].to(device)
            data = torch.cat((data, poi_sample), 0)
            target = torch.cat((target, poi_sample_target), 0)
            print("***************", data.shape[0], target.shape[0])

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        train_running_loss += loss.item()
        _, preds = torch.max(output.data, 1)
        train_running_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()
        print('Batch Acc:', (preds == target).sum().item()/len(data))
    train_loss = train_running_loss / len(train_dataloader.dataset)
    train_accuracy = 100. * train_running_correct / len(poi_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    return train_loss, train_accuracy

def poitrain(model, optimizer, train_dataloader, poi_loader, test_dataloader, device, model_dir, args):
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    attack_loss, attack_accraucy = [], []
    best_acc = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        print("current learning rate", optimizer.param_groups[0]['lr'])
        train_epoch_loss, train_epoch_accuracy = fit(model, optimizer, train_dataloader, poi_loader, device)
        val_epoch_loss, val_epoch_accuracy = validate(model, test_dataloader, device)
        attack_epoch_loss, attack_epoch_accuracy = validate_attack(model, test_dataloader, device, args)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        attack_loss.append(attack_epoch_loss)
        attack_accraucy.append(attack_epoch_accuracy)
        scheduler.step()
        # save model
        if args.save and val_accuracy[-1] >= best_acc:
            best_acc = val_accuracy[-1]
            filename = os.path.join(model_dir,
                                    args.dataset + '_acc' + str(round(val_accuracy[-1], 2)) +
                                    '_atk' + str(round(attack_accraucy[-1], 2)) + '%_params.pt')
            torch.save(model.state_dict(), filename)

