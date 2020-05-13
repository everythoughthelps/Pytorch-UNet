import os
import torch
from torch import nn
from torchvision.transforms import ToPILImage
import numpy as np
import torch.nn.functional as F
def eval_net(net, dataset, epoch, classes,args,gpu=True):
    global val_rmse, mask_pred_sparse,  best_threshold_val_rmse
    net.eval()
    total_rmse = 0
    for i, b in enumerate(dataset):
        img = b['image']
        img = img.float()
        depth = b ['depth']
        if gpu:
            img = img.cuda()
            depth = depth.cuda()

        out = net(img)

        #depth = torch.range(1, 64)
        #depth = depth.unsqueeze(0)
        #depth = depth.unsqueeze(2)
        #depth = depth.unsqueeze(2)

        mask_prob = F.softmax(out, dim=1)
        #_,mask_pred_sparse = mask_prob.max(dim=1)
        mask_pred_sparse = soft_sum(mask_prob,args)
        #true_mask = true_mask * (256//classes)

        loss = nn.MSELoss()
        mse = loss(mask_pred_sparse.float(),depth).item()
        rmse = np.sqrt(mse)
        total_rmse += rmse
        print(i/len(dataset),b['image_name'] ,rmse)
        break

    val_rmse = total_rmse / len(dataset)

    if epoch == 0:
        best_threshold_val_rmse = val_rmse

    if val_rmse < best_threshold_val_rmse:
        best_threshold_val_rmse = val_rmse

    if epoch % 20 ==0 or val_rmse == best_threshold_val_rmse:
        print('save results')
        for j,m in enumerate(dataset):
            img = m['image']
            img = img.float()

            if gpu:
                img = img.cuda()

            out = net(img)
            #depth = torch.range(1, 64)
            #depth = depth.unsqueeze(0)
            #depth = depth.unsqueeze(2)
            #depth = depth.unsqueeze(2)

            mask_prob = F.softmax(out, dim=1)
            #_,mask_pred_sparse = mask_prob.max(dim=1)
            #mask_pred_sparse = mask_pred_sparse * (256//classes)
            mask_pred_sparse = soft_sum(mask_prob,args)
            results_imgs = ToPILImage()(mask_pred_sparse.squeeze().float().cpu() / 10)
            if not os.path.exists('results'):
                os.mkdir('results')
            if not os.path.exists('results/' + str(epoch) + 'epochs_results'):
                os.mkdir('results/' + str(epoch) + 'epochs_results')
            results_imgs.save(os.getcwd() +'/results/' + str(epoch) + 'epochs_results/' + str(m['image_name']).strip(str(['data/nyu2_test/.png']))+'.png')
            break
    return val_rmse

def soft_sum(probs,args):
    ones = torch.ones(probs.size()).float().cuda()
    unit = torch.arange(0, args.num_classes).view(1, probs.size(1), 1, 1).float()
    weight = unit
    for _ in range(probs.size(0) - 1):
        weight = torch.cat((weight, unit), dim=0)
    weight = ones * weight.cuda()
    q = (np.log10(10) - np.log10(args.e)) / (args.num_classes - 1)
    weight = weight * q + np.log10(args.e)
    depth_value = 10 ** (torch.sum(weight * probs, dim=1)) - args.e
    depth_value = torch.unsqueeze(depth_value, dim=1)

    #q = (np.log10(10+args.e) - np.log10(args.e)) / (args.num_classes - 1)
    #_,label = probs.max(dim = 1)
    #lgdepth = label * q + np.log10(args.e)
    #depth_value = 10 ** (lgdepth) - args.e
    #depth_value = torch.unsqueeze(depth_value, dim=1)
    return depth_value
