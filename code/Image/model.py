import torch.nn as nn
import torch.optim as optim
import torchvision.models as models


def Resnet18(class_num, args):
    resnet18 = models.resnet18(pretrained=True)
    num_ftrs = resnet18.fc.in_features
    #for param in resnet18.parameters():
    #    param.requires_grad = False
    resnet18.fc = nn.Linear(num_ftrs, class_num)
    optimizer = optim.SGD(resnet18.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return resnet18, optimizer

models.vgg

def VGG16(class_num, args):
    vgg16 = models.vgg16_bn(pretrained=True)
    print(vgg16)
    num_ftrs = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_ftrs, class_num)
    #for param in vgg16.features.parameters():
    #    param.requires_grad = False
    optimizer = optim.SGD(vgg16.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return vgg16, optimizer

def Inception(class_num, args):
    inception = models.inception_v3(pretrained=True)
    inception.classifier[6].out_features = class_num
    for param in inception.features.parameters():
        param.requires_grad = False
    optimizer = optim.SGD(inception.classifier.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    return inception, optimizer