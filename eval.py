import torch
import torch.nn.functional as F
from torch import nn

from dice_loss import dice_coeff


def eval_net(net, dataset, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    global i
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        true_mask = b[1]

        img = torch.from_numpy(img).unsqueeze(0)
        true_mask = torch.from_numpy(true_mask).unsqueeze(0)

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        mask_pred = net(img)
        print(mask_pred.size())
        probability,mask_pred= torch.max(mask_pred,dim=1)
        mask_pred = mask_pred
        print(mask_pred.size())
        loss = nn.MSELoss()
        tot += loss(mask_pred,true_mask)
        break
    return tot / (i + 1)

