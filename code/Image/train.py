import os
import torch
import torch.nn as nn
from utils import validate
from tqdm import tqdm

def fit(model, optimizer, train_dataloader, device):
    model.train()
    train_running_loss = 0.0
    train_running_correct = 0
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(tqdm(train_dataloader)):
        data, target = data[0].to(device), data[1].to(device)
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
    train_accuracy = 100. * train_running_correct / len(train_dataloader.dataset)
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}')
    return train_loss, train_accuracy

def train(model, optimizer, train_dataloader, test_dataloader, device, model_dir, args):
    train_loss, train_accuracy = [], []
    val_loss, val_accuracy = [], []
    best_acc = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    for epoch in range(args.epochs):
        train_epoch_loss, train_epoch_accuracy = fit(model, optimizer, train_dataloader, device)
        val_epoch_loss, val_epoch_accuracy = validate(model, test_dataloader, device)
        train_loss.append(train_epoch_loss)
        train_accuracy.append(train_epoch_accuracy)
        val_loss.append(val_epoch_loss)
        val_accuracy.append(val_epoch_accuracy)
        scheduler.step()
        # save model
        if args.save and val_accuracy[-1] >= best_acc:
            best_acc = val_accuracy[-1]
            filename = os.path.join(model_dir, args.dataset + '_' + str(round(val_accuracy[-1], 2)) + '%_params.pt')
            torch.save(model.state_dict(), filename)

