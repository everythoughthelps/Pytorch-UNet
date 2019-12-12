import sys
import numpy as np
import os
import time
from optparse import OptionParser
import visdom
import logging
import torch
import torch.nn as nn
from utils.NYU_dataloader import nyudataset
from torch.utils.data import DataLoader
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from eval import eval_net
from unet.unet_model import UNet
from unet.unet_model import hopenet
import torchvision.models.resnet as rn

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = [0]

vis=visdom.Visdom(env='test1')
vis.line([[0.,0.]], [0], win='train', opts=dict(title='loss&acc', legend=['loss', 'acc']))
def train_net(net,epochs=5,batchsize=5,lr=0.1,save_cp=True,gpu=True):

    dir_checkpoint = 'checkpoints/'
    global i, mask_sparse, imgs, depth, best_val_rmse, masks_prob, masks_pred
    dir_img = '/home/panmeng/data/nyu_images/train_dir'
    dir_mask = '/home/panmeng/data/nyu_depths/train_dir'
    val_img_dir = '/home/panmeng/data/nyu_images/test_dir/'
    val_mask_dir = '/home/panmeng/data/nyu_depths/test_dir/'
    train_dataset = nyudataset(dir_img,dir_mask,scale=0.1)
    val_dataset = nyudataset(val_img_dir,val_mask_dir,scale=0.1)
    train_dataloader = DataLoader(train_dataset,batch_size=batchsize,shuffle=False)
    test_dataloader = DataLoader(val_dataset,batch_size=1,shuffle=False)

    print('''
    Starting training:
        Epochs: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs,lr, len(train_dataset),
               len(val_dataset), str(save_cp), str(gpu)))

    N_train = len(train_dataloader)
    optimizer = optim.SGD(net.parameters(),
                          lr=lr,
                          momentum=0.9,
                          weight_decay=0.0005)
    scheduler = MultiStepLR(optimizer,[80,1000,1500],gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    criterion2 = nn.MSELoss()
    x_list = []
    y_crossentropy_list = []
    y_RMSE_list = []


    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        print(scheduler.get_lr())
        net.train()

        epoch_loss = 0
        start_time = time.time()
        for i, b in enumerate(train_dataloader):

            imgs = b[0]
            imgs = imgs.float()
            mask_sparse = b[1]
            mask_sparse = mask_sparse.float()

            imgs = imgs.cuda()
            mask_sparse = mask_sparse.cuda()

            out = net(imgs)
            loss_crossentropy = criterion(out,mask_sparse.long())

            #out_mse = F.softmax(out,dim=1)
            #depth = torch.range(1,64)
            #depth = depth.unsqueeze(0)
            #depth = depth.unsqueeze(2)
            #depth = depth.unsqueeze(2)

            #masks_pred = torch.mul(out_mse,depth.cuda())
            masks_prob = torch.softmax(out,dim=1)
            _,masks_pred = torch.max(masks_prob,dim=1)
            loss_mse = criterion2(masks_pred.float(), mask_sparse.float())
            loss = loss_mse + loss_crossentropy
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('time',(time.time()-start_time)/60)
        scheduler.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / (i+1)))

        with open('train_log.txt','a') as f:
            f.write(str(epoch_loss /(i+1)) +'\n')

        x_list.append(epoch)
        y_crossentropy_list.append(str(epoch_loss/(i+1))+'\n')

        del mask_sparse,masks_prob,imgs,masks_pred

        val_rmse= eval_net(net, test_dataloader,epoch,gpu=True)
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

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=2000, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=1,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.01,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')#'/home/panmeng/PycharmProjects/pt1.1/unet/checkpoints/CP1.pth'
    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net =hopenet(rn.Bottleneck, [3, 4, 6, 3], 64)
    print(net)
    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))
    if args.gpu:
        net.cuda()
        net = nn.DataParallel(net,device_ids=gpus)
    print(torch.cuda.is_available())
        # cudnn.benchmark = True # faster convolutions, but more memory


    train_net(net=net,
              epochs=args.epochs,
              batchsize=args.batchsize,
              lr=args.lr,
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
