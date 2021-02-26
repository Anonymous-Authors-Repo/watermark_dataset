import torch
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def pgd_attack(model, images, ground_label, eps=16/255, alpha=4/255, iters=40):
    loss = torch.nn.CrossEntropyLoss()
    images = images.to(device)
    ori_image = images
    ground_label = ground_label.to(device).type(torch.long)
    ori_images = images.data
    for i in range(iters):
        images.requires_grad = True
        outputs = model(images)

        model.zero_grad()
        cost = loss(outputs, ground_label)
        cost.backward()

        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = torch.clamp(ori_images + eta, min=0, max=1).detach_()
    print("*************** adversarial samples***************")
    print('Ori', torch.max(model(ori_images).data, 1).indices)
    print('atk', torch.max(model(images).data, 1).indices)
    print('grd', ground_label.data)
    return images