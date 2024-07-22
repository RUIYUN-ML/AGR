# The source code is from: https://github.com/YisenWang/MART
import torch
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F

def mart_loss(model,
              img,
              img_adv,
              target,
              optimizer,
              device,
              beta=6.0,
              ):

    kl = nn.KLDivLoss(reduction='none').to(device)

    x = img.clone().to(device).detach()
    x_adv = img_adv.clone().to(device).detach()
    y = target.clone().to(device).detach()
    batch_size = len(x)
    
    optimizer.zero_grad()

    logits = model(x)

    logits_adv = model(x_adv)

    adv_probs = F.softmax(logits_adv, dim=1)

    tmp1 = torch.argsort(adv_probs, dim=1)[:, -2:]

    new_y = torch.where(tmp1[:, -1] == y, tmp1[:, -2], tmp1[:, -1])

    loss_adv = F.cross_entropy(logits_adv, y).to(device) + F.nll_loss(torch.log(1.0001 - adv_probs + 1e-12), new_y).to(device)

    nat_probs = F.softmax(logits, dim=1)

    true_probs = torch.gather(nat_probs, 1, (y.unsqueeze(1)).long()).squeeze()

    loss_robust = (1.0 / batch_size) * torch.sum(
        torch.sum(kl(torch.log(adv_probs + 1e-12), nat_probs), dim=1) * (1.0000001 - true_probs))

    return loss_adv, beta * loss_robust