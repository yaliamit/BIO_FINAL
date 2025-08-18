from model.layers import *
import torch

class Unet(nn.Module):
    def __init__(self, n_channels, n_classes, kernel_size=3, padding=1, bilinear=False, n_layers = 4, LI=0):
        super(Unet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.n_layers = n_layers

        assert 2 <= n_layers <= 4, "n_layers must be between 2 and 4"
        factor = 2 if bilinear else 1

        self.inc = (DoubleConv(n_channels, 64, kernel_size, padding))
        self.down1 = (Down(64, 128, kernel_size, padding))
        self.down2 = (Down(128, 256, kernel_size, padding))
        ff=256
        # Adjust number of layers based on parameter
        if n_layers > 2: 
            self.down3 = (Down(256, 512, kernel_size, padding))
            ff=512
        if n_layers > 3:
            self.down4 = (Down(512, 1024 // factor, kernel_size, padding))
            ff=1024
        self.LI=LI
        print('LI',self.LI)
        # Up layers change channel sizes based on how many down layers we have
        if n_layers == 4: 
            self.up1 = (Up(1024, 512 // factor, kernel_size, padding, bilinear))
            self.up2 = (Up(512, 256 // factor, kernel_size, padding, bilinear))
            self.up3 = (Up(256, 128 // factor, kernel_size, padding, bilinear))
            self.up4 = (Up(128, 64, kernel_size, padding, bilinear))
        elif n_layers == 3: 
            self.up1 = Up(512, 256 // factor, kernel_size, padding, bilinear)
            self.up2 = Up(256, 128 // factor, kernel_size, padding, bilinear)
            self.up3 = Up(128, 64, kernel_size, padding, bilinear)
        elif n_layers == 2:
            self.up1 = Up(256, 128 // factor, kernel_size, padding, bilinear)
            self.up2 = Up(128, 64, kernel_size, padding, bilinear)
            
        self.outc = (OutConv(64, n_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)    
        if self.n_layers == 3:
            x4 = self.down3(x3)
           
            x = self.up1(x4, x3)
            x = self.up2(x, x2)
            x = self.up3(x, x1)
        if self.n_layers == 4:
            x4 = self.down3(x3)
            x5 = self.down4(x4)
            # if self.SA:
            #     x5=self.SA(x5)
            x = self.up1(x5, x4)
            x = self.up2(x, x3)
            x = self.up3(x, x2)
            x = self.up4(x, x1)
        if self.n_layers == 2:
            # if self.SA:
            #     x3=self.SA(x3)
            x = self.up1(x3, x2)
            x = self.up2(x, x1)
        logits = self.outc(x)
        
        if str(self.LI)=='True':
            self.LI=2
        
        if (self.n_classes == 1 and self.LI==0) or self.LI==2:
            return self.sigmoid(logits)
        elif self.LI==1:
            return(logits)
        else:
            return logits

class UnetRes(nn.Module):
    def __init__(self, n_channels, n_classes, kernel_size=3, padding=1, bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.inc = (ResConv(n_channels, 64, kernel_size, padding))
        self.down1 = (DownRes(64, 128, kernel_size, padding))
        self.down2 = (DownRes(128, 256, kernel_size, padding))
        self.down3 = (DownRes(256, 512, kernel_size, padding))
        factor = 2 if bilinear else 1
        self.down4 = (DownRes(512, 1024 // factor, kernel_size, padding))
        self.up1 = (UpRes(1024, 512 // factor, kernel_size, padding, bilinear))
        self.up2 = (UpRes(512, 256 // factor, kernel_size, padding, bilinear))
        self.up3 = (UpRes(256, 128 // factor, kernel_size, padding, bilinear))
        self.up4 = (UpRes(128, 64, kernel_size, padding, bilinear))
        self.outc = (OutConv(64, n_classes))
        self.sigmoid = nn.Sigmoid()
        
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
        logits = self.outc(x)
        
        if self.n_classes == 1:
            return self.sigmoid(logits)
        else:
           
            return logits

