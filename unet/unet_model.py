# full assembly of the sub-parts to form the complete net

from torchvision.models import  resnet as rn
import torch.nn.functional as F
from .unet_parts import *

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, n_classes)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

class hopenet(rn.ResNet):

    def __init__(self, block, layers,n_classes):
        super(hopenet,self).__init__(block,layers)
        self.inplanes = 128
        self.conv1 = nn.Conv2d(3,128,kernel_size=7,padding=3,stride=2)
        self.bn1 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block,64,layers[0])
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)
        self.up1 = up(2048, 1024)
        self.up2 = up(1024, 512)
        self.up3 = up(512, 256)
        self.up4 = up(256, 128)
        self.outc = outconv(128, n_classes)


    def forward(self,x):
        print(x.size())
        x = self.conv1(x)
        print(x.size())
        x = self.bn1(x)
        x1 = self.relu(x)

        x2 = self.layer1(x1)
        print(x2.size())
        x3 = self.layer2(x2)
        print(x3.size())
        x4 = self.layer3(x3)
        print(x4.size())
        x5 = self.layer4(x4)
        print(x5.size())
        x = self.up1(x5, x4)
        print(x.size())
        x = self.up2(x, x3)
        print(x.size())
        x = self.up3(x, x2)
        print(x.size())
        x = self.up4(x, x1)
        print(x.size())
        x = self.outc(x)
        print(x.size())
        return x
