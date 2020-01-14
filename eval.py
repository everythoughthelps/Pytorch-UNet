import os
import torch
from torch import nn
from torchvision.transforms import ToPILImage
import numpy as np
import torch.nn.functional as F
def eval_net(net, dataset, epoch, classes,gpu=True):
    global val_rmse, mask_pred_sparse,  best_threshold_val_rmse
    net.eval()
    total_rmse = 0
    for i, b in enumerate(dataset):
        img = b[0]
        img = img.float()
        true_mask = b[1]
        true_mask = true_mask.float()
        if gpu:
            img = img.cuda()
            true_mask = true_mask.cuda()

        out = net(img)

        #depth = torch.range(1, 64)
        #depth = depth.unsqueeze(0)
        #depth = depth.unsqueeze(2)
        #depth = depth.unsqueeze(2)

        mask_prob = F.softmax(out, dim=1)
        _,mask_pred_sparse = mask_prob.max(dim=1)
        mask_pred_sparse = mask_pred_sparse * (256//classes)
        true_mask = true_mask * (256//classes)

        loss = nn.MSELoss()
        mse = loss(mask_pred_sparse.float(),true_mask).item()
        rmse = np.sqrt(mse)
        total_rmse += rmse
        print(i/len(dataset),b[2] ,rmse)
        break

    val_rmse = total_rmse #/ len(dataset)

    if epoch == 0:
        best_threshold_val_rmse = val_rmse

    if val_rmse < best_threshold_val_rmse:
        best_threshold_val_rmse = val_rmse

    if epoch % 20 ==0 or val_rmse == best_threshold_val_rmse:
        print('save results')
        for j,m in enumerate(dataset):
            img = m[0]
            img = img.float()

            if gpu:
                img = img.cuda()

            out = net(img)
            #depth = torch.range(1, 64)
            #depth = depth.unsqueeze(0)
            #depth = depth.unsqueeze(2)
            #depth = depth.unsqueeze(2)

            mask_prob = F.softmax(out, dim=1)
            _,mask_pred_sparse = mask_prob.max(dim=1)
            mask_pred_sparse = mask_pred_sparse * (256//classes)

            results_imgs = ToPILImage()(mask_pred_sparse.float().cpu() / 255)
            if not os.path.exists('results'):
                os.mkdir('results')
            if not os.path.exists('results/' + str(epoch) + 'epochs_results'):
                os.mkdir('results/' + str(epoch) + 'epochs_results')
            results_imgs.save(os.getcwd() +'/results/' + str(epoch) + 'epochs_results/' + str(m[2]).strip('(),\''))
            break
    return val_rmse
