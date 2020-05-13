import sys
import numpy as np
import os
import time
from optparse import OptionParser
import visdom
import logging
import torch
import torch.nn as nn
from utils import loaddata
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import utils
from eval import eval_net
from unet.unet_model import UNet
from unet.unet_model import hopenet
import torch.nn.functional as F
import torchvision.models.resnet as rn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = [0]

def get_args():
    parser = OptionParser()
    parser.add_option('--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('--e', default=0.01, type=float, help='avoid log0')
    parser.add_option('--batch_size',  default=2,type='int', help='batch size')
    parser.add_option('-r', '--learning-rate', dest='lr', default=0.1,type='float', help='learning rate')
    parser.add_option('--data', default='/data/nyuv2/', type=str, help='path of dataset')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-l', '--load', dest='load',
                      default=False, help='load file model')#'/home/panmeng/PycharmProjects/pt1.1/unet/checkpoints/CP1.pth'
    parser.add_option('--num_classes', default=120)
    (options, args) = parser.parse_args()
    return options

def train_net(net,epochs,lr,classes,save_cp=True,gpu=True,):

    dir_checkpoint = 'checkpoints/'
    global i, mask_sparse, imgs, depth, best_val_rmse, masks_prob, masks_pred

    train_loader = loaddata.getTrainingData(args)
    test_loader = loaddata.getTestingData(args)
    print('''
    Starting training:
        Epochs: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs,lr, len(train_loader),
               len(test_loader), str(save_cp), str(gpu)))

    N_train = len(train_loader)
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer,[10,25],gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    x_list = []
    y_crossentropy_list = []
    y_RMSE_list = []


    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print(scheduler.get_lr())
        net.train()

        epoch_loss = 0
        start_time = time.time()
        for i, b in enumerate(train_loader):

            imgs = b['image']
            imgs = imgs.float()
            print(imgs.size())
            mask_sparse = b['label']
            mask_sparse = mask_sparse.float()
            print(mask_sparse.size())


            imgs = imgs.cuda()
            mask_sparse = mask_sparse.cuda()

            out = net(imgs)
            loss_crossentropy = criterion(out,mask_sparse.long().cuda())

            #out_mse = F.softmax(out,dim=1)
            #depth = torch.range(1,64)
            #depth = depth.unsqueeze(0)
            #depth = depth.unsqueeze(2)
            #depth = depth.unsqueeze(2)

            #masks_pred = torch.mul(out_mse,depth.cuda())
            loss = loss_crossentropy
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            break
        print('time',(time.time()-start_time)/60)
        scheduler.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (i+1)))

        with open('train_log.txt','a') as f:
            f.write(str(epoch_loss /(i+1)) +'\n')

        x_list.append(epoch)
        y_crossentropy_list.append(str(epoch_loss/(i+1))+'\n')


        val_rmse= eval_net(net, test_loader,epoch,classes,args,gpu=True)
        print('Validation RMSE: {}'.format(val_rmse))

        with open('val_log.txt','a') as f:
            f.write(str(val_rmse) +'\n')
        y_RMSE_list.append(str(val_rmse)+'\n')
        vis.line([[np.array(epoch_loss/(i+1)),np.array(val_rmse)]], [np.array(epoch)], win='train', update='append')
        time.sleep(0.5)

        if epoch == 0:
            best_val_rmse = val_rmse

        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            if not os.path.exists('checkpoints'):
                os.mkdir('checkpoints')
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


if __name__ == '__main__':
    args = get_args()

    net =hopenet(rn.Bottleneck, [3, 4, 6, 3],args.num_classes)
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    if args.gpu:
        net.cuda()
        net = nn.DataParallel(net,device_ids=gpus)
    print(torch.cuda.is_available())
        # cudnn.benchmark = True # faster convolutions, but more memory

    vis = visdom.Visdom(env='test1')
    vis.line([[0., 0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))

    train_net(net=net,
              epochs=args.epochs,
              lr=args.lr,
              classes=args.num_classes,
              gpu=args.gpu)

    try:
        sys.exit(0)
    except SystemExit:
        os._exit(0)
'''
LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(pathname)s %(message)s "#配置输出日志格式
DATE_FORMAT = '%Y-%m-%d  %H:%M:%S %a ' #配置输出时间的格式，注意月份和天数不要搞乱了
logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT,
                    datefmt = DATE_FORMAT ,
                    filename=r"d:\test\test.log" #有了filename参数就不会直接输出显示到控制台，而是直接写入文件
                    )
logging.debug("msg1")
logging.info("msg2")
logging.warning("msg3")
logging.error("msg4")
logging.critical("msg5")
'''
