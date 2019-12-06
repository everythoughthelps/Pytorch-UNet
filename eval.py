import os
import torch
from torch import nn
from torchvision.transforms import ToPILImage
import numpy as np

def eval_net(net, dataset, epochs,best_threshold_val_rmse, gpu=True):
    global i, val_rmse
    net.eval()
    tot = 0
    for i, b in enumerate(dataset):
        img = b[0]
        img = img.float()
        true_mask = b[1]
        true_mask = true_mask.float()

        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        pred = net(img)

        depth = torch.range(1, 64)
        depth = depth.unsqueeze(0)
        depth = depth.unsqueeze(2)
        depth = depth.unsqueeze(2)

        mask_pred_sparse= torch.mul(pred,depth.cuda())
        mask_pred_sparse = mask_pred_sparse.float()
        mask_pred_sparse = mask_pred_sparse * 4
        true_mask = true_mask * 4

        loss = nn.MSELoss()
        tot += loss(mask_pred_sparse,true_mask)

    val_mse = tot /(i + 1)
    val_rmse = torch.sqrt(val_mse)

    if val_rmse < best_threshold_val_rmse:
        best_threshold_val_rmse = val_rmse

    if epochs % 20 ==0 or val_rmse == best_threshold_val_rmse:
        print('save results')

        for j,m in enumerate(dataset):
            img = m[0]
            img = img.float()

            if gpu:
                img = img.cuda()

            pred = net(img)
            probability, mask_pred_sparse = torch.max(pred, dim=1)

            imgs = ToPILImage()(mask_pred_sparse.float().cpu() * 4)

            if not os.path.exists('results'):
                os.mkdir('results')
            if not os.path.exists('results/'+str(epochs)+'epochs_results'):
                os.mkdir('results/'+str(epochs)+'epochs_results')
            imgs.save(os.getcwd()+'/results/'+str(epochs)+'epochs_results/'+str(j)+'.png')
    return val_rmse
