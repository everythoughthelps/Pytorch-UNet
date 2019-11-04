import sys
import os
from optparse import OptionParser
import numpy as np
import logging
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from matplotlib import pyplot as plt
from torch.optim.lr_scheduler import StepLR

from eval import eval_net
from unet.unet_model import UNet
from utils import get_ids, split_train_val, get_imgs_and_masks, batch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = [0]
def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=True,
              img_scale=1):

    global i, masks_pred, true_masks
    dir_img = '/home/panmeng/data/nyu_images/'
    dir_mask = '/home/panmeng/data/nyu_depths/'
    dir_checkpoint = 'checkpoints/'
    print(lr)
    ids = get_ids(dir_img)

    iddataset = split_train_val(ids, val_percent)
    print(iddataset)
    print(type(iddataset))
    print(iddataset['train'])
    print(iddataset['val'])

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(iddataset['train']),
               len(iddataset['val']), str(save_cp), str(gpu)))

    N_train = len(iddataset['train'])
    print(N_train)
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    scheduler = StepLR(optimizer,step_size=10,gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    #criterion2 = nn.MSELoss()
    x_list = []
    y_crossentropy_list = []
    y_RMSE_list = []
    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()
        print(lr)
        # reset the generators
        train =get_imgs_and_masks(iddataset['train'], dir_img, dir_mask,scale=1)
        val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask,scale=1)

        epoch_loss = 0
        for i, b in enumerate(batch(train, batch_size)):
            imgs = np.array([i[0] for i in b]).astype(np.float32)
            true_masks = np.array([i[1] for i in b]).astype(np.float32)

            imgs = torch.from_numpy(imgs)
            true_masks = torch.from_numpy(true_masks)
            imgs = imgs.cuda()
            true_masks = true_masks.cuda()

            masks_pred = net(imgs)

            print(masks_pred.size(),masks_pred.dtype)
            print(true_masks.size(),true_masks.dtype)

            loss = criterion(masks_pred, true_masks)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            break


        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        with open('train_log.txt','a') as f:
            f.write(str(epoch_loss /i) +'\n')

        x_list.append(epoch)
        y_crossentropy_list.append(str(epoch_loss/i)+'\n')

        del true_masks,masks_pred

        print(val,type(val))

        val_MSE= eval_net(net, val, gpu)
        val_RMSE= torch.sqrt(val_MSE)
        print('Validation crossentropy: {}'.format(val_RMSE))

        with open('val_log.txt','a') as f:
            f.write(str(val_RMSE) +'\n')
        y_RMSE_list.append(str(val_RMSE)+'\n')

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
    with open('log1.txt','w') as f:
        f.write(str(y_crossentropy_list))
        f.write(str(y_RMSE_list))
    #plt.title('300 epoch train loss and validate coeff')
    #plt.ylabel('accuracy')
    #plt.xlabel('epoch')
    #plt.plot(x_list, y_loss_list,label='train loss',color='green')
    #plt.plot(x_list,y_coeff_list,label='val coeff',color='red')
    #plt.savefig("300 epoch train loss and validate coeff")
    #plt.show()



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=400, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.2,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')#'/home/panmeng/PycharmProjects/pt1.1/unet/checkpoints/CP1.pth'
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=256)
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    if args.gpu:
        net.cuda()
        net = nn.DataParallel(net,device_ids=gpus)
    print(torch.cuda.is_available())
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
