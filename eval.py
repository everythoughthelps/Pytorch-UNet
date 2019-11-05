import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import ToPILImage


def eval_net(net, dataset, epochs,best_threshold_val_rmse, gpu=True):
    global i, val_rmse
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

        pred = net(img)
        probability,mask_pred= torch.max(pred,dim=1)
        mask_pred = mask_pred.float()
        loss = nn.MSELoss()
        tot += loss(mask_pred,true_mask)

        val_mse = tot /(i + 1)
        val_rmse = torch.sqrt(val_mse)

        if val_rmse < best_threshold_val_rmse:
            best_threshold_val_rmse = val_rmse

        if epochs % 20 ==0 or val_rmse == best_threshold_val_rmse:

            imgs = ToPILImage()(mask_pred.cpu())

            if not os.path.exists('results'):
                os.mkdir('results')
            if not os.path.exists('results/'+str(epochs)+'epochs_results'):
                os.mkdir('results/'+str(epochs)+'epochs_results')
            imgs.save(os.getcwd()+'/results/'+str(epochs)+'epochs_results/'+str(i)+'.png')
    return val_rmse

