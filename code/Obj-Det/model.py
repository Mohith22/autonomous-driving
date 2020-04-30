import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import torchvision.models as models

# ---------------- SELF SUPERVISED CLASSIFIER ---------------- #
class BasicClassifierSSL(nn.Module):
    def __init__(self):
        super(BasicClassifierSSL, self).__init__()
        self.encoder_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=(2,3)),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 256 x 308
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 64 x 128 x 154
            nn.Conv2d(64, 192, kernel_size=3, padding=(1,2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(192),
            #Current Size:- 192 x 128 x 156
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 384 x 128 x 156
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            #Current Size:- 256 x 128 x 156
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 256 x 64 x 78
            nn.Conv2d(256, 192, kernel_size=(3,5), padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 192 x 64 x 76
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            #Current Size:- 64 x 64 x 76
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 32 x 64 x 76
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 32 x 32 x 38
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 1 x 32 x 38

        )
        self.fc1 = nn.Linear(32*38, 6)

        
    def forward(self,x):
        x = self.encoder_features(x)
        x = x.view(-1, 32*38)
        x = self.fc1(x)
        return x


# ---------------- MINI ENCODER DECODER ---------------- #
class Mini_Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Mini_Encoder_Decoder, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoders = nn.ModuleList()
        for _ in range(6):
            self.encoders.append(Mini_Encoder())
        self.decoder = Mini_Decoder()

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoders[i](x[i]))
        encoder_output = torch.stack(encoder_outs).permute(0,2,1,3,4)
        encoder_output = torch.cat([i for i in encoder_output]).permute(1,0,2,3)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

# ---------------- MINI ENCODER ---------------- #
class Mini_Encoder(nn.Module):
    def __init__(self):
        super(Mini_Encoder, self).__init__()
        self.encoder_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=(2,3)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            #Current Size:- 64 x 256 x 308
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 64 x 128 x 154
            nn.Conv2d(64, 128, kernel_size=3, padding=(1,2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            #Current Size:- 192 x 128 x 156
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            #Current Size:- 64 x 128 x 156
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #Current Size:- 32 x 128 x 156
        )
        
    def forward(self,x):
        return self.encoder_features(x)

# ---------------- MINI DECODER ----------------  #
class Mini_Decoder(nn.Module):
    def __init__(self):
        super(Mini_Decoder, self).__init__()
        self.decoder_features = nn.Sequential(
            nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True),
            #Current Size:- 32 x 100 x 100
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 16 x 200 x 200
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            #Current Size:- 8 x 400 x 400
            nn.Conv2d(32, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 4 x 400 x 400
            nn.ConvTranspose2d(8, 1,kernel_size=4, stride=2, padding=1),
            #nn.ReLU(inplace=True),
            nn.Sigmoid(),
            #Current Size:- 1 x 800 x 800
        )
        
    def forward(self,x):
        return self.decoder_features(x)

# ---------------- Modified Resnet ENCODER ---------------- #
class Resnet_Encoder_Decoder(nn.Module):
    def __init__(self, use_bce=False):
        super(Resnet_Encoder_Decoder, self).__init__()
        self.encoders = nn.ModuleList()
        for _ in range(6):
            self.encoders.append(Resnet_Encoder())
        self.decoder = Resnet_Decoder()
        self.sigmoid = nn.Sigmoid()
        self.use_bce = use_bce
          
    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoders[i](x[i]))
        encoder_output = torch.stack(encoder_outs).permute(0,2,1,3,4)
        encoder_output = torch.cat([i for i in encoder_output]).permute(1,0,2,3)
        decoder_output = self.decoder(encoder_output)
        if (not self.use_bce):
            decoder_output = self.sigmoid(decoder_output) 
        return decoder_output

class Resnet_Encoder(nn.Module):
    def __init__(self):
        super(Resnet_Encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=False)
        layers = list(resnet18.children())[0:4]
        self.encoder_features = nn.Sequential(*layers)
    def forward(self,x):
        return self.encoder_features(x)

class Resnet_Decoder(nn.Module):
    def __init__(self):
        super(Resnet_Decoder, self).__init__()
        #Size:- 64 x 56 x 56
        self.decoder_features = nn.Sequential(

            nn.Upsample(size=(100, 100), mode='bilinear', align_corners=True),

            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(192,192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192,192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(96,96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96,96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(48,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),

            #Current Size:- 1 x 800 x 800
        )
        
    def forward(self,x):
        return self.decoder_features(x)
        
# ---------------- SPATIAL ENCODER DECODER ---------------- #
class Spatial_Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Spatial_Encoder_Decoder, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoders = nn.ModuleList()
        for _ in range(6):
            self.encoders.append(SEncoder())
        self.decoder = Spatial_Decoder()

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        x = ((x-x.min())/(x.max()-x.min())) - 0.5
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoders[i](x[i]))
        top_enc = torch.cat((encoder_outs[0], encoder_outs[1], encoder_outs[2]), dim=3)
        top_dec = torch.cat((encoder_outs[3], encoder_outs[4], encoder_outs[5]), dim=3)
        encoder_output = torch.cat((top_enc, top_dec), dim=2)
        #print(encoder_output.size()) - #Size - torch.Size([2, 32, 256, 468])
        decoder_output = self.decoder(encoder_output)
        return decoder_output

# ---------------- SPATIAL DECODER ---------------- #
class Spatial_Decoder(nn.Module):
    def __init__(self):
        super(Spatial_Decoder, self).__init__()
        self.decoder_features = nn.Sequential(
            nn.Upsample(size=(400, 400), mode='bilinear', align_corners=True),
            #Current Size:- 32 x 400 x 400
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 400 x 400
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            #Current Size:- 8 x 800 x 800
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            #nn.ReLU(inplace=True),
            #Current Size:- 2 x 800 x 800
            
            #nn.ConvTranspose2d(8, 2,kernel_size=4, stride=2, padding=1),
            #nn.ReLU(inplace=True),
            #Current Size:- 2 x 800 x 800
        )
        
    def forward(self,x):
        return self.decoder_features(x)

# ---------------- SIAMEESE KIND OF NETWORK (NOT WORKING!) ---------------- #
#Siameese Single Encoder 
class Single_Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Single_Encoder_Decoder, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoder_features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, padding=(2,3)),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 256 x 308
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 64 x 128 x 154
            nn.Conv2d(64, 192, kernel_size=3, padding=(1,2)),
            nn.ReLU(inplace=True),
            #Current Size:- 192 x 128 x 156
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 384 x 128 x 156
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 256 x 128 x 156
            nn.MaxPool2d(kernel_size=2, stride=2),
            #Current Size:- 256 x 64 x 78
            nn.Conv2d(256, 192, kernel_size=(3,5), padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 192 x 64 x 76
            nn.Conv2d(192, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 64 x 76
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 64 x 76
        )

        self.decoder_features = nn.Sequential(
        	nn.Upsample(size=(100,100), mode='bilinear', align_corners=True),
        	#Current Size:- 384 x 100 x 100
        	nn.ConvTranspose2d(384, 256, kernel_size=4, stride=2, padding=1),
        	nn.ReLU(inplace=True),
        	#Current Size:- 256 x 200 x 200
        	nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        	#Current Size:- 192 x 200 x 200
        	nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1),
        	nn.ReLU(inplace=True),
        	#Current Size:- 64 x 400 x 400
        	nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 32 x 400 x 400
        	nn.ConvTranspose2d(32, 1,kernel_size=4, stride=2, padding=1),
        	nn.Sigmoid(),
        	#Current Size:- 1 x 800 x 800
        )

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoder_features(x[i]))
        encoder_output = torch.stack(encoder_outs).permute(0,2,1,3,4)
        encoder_output = torch.cat([i for i in encoder_output]).permute(1,0,2,3)
        decoder_output = self.decoder_features(encoder_output)
        return decoder_output

# ---------------- UNET ENCODER AND VANILLA DECODER ---------------- #
class UNet_Encoder_Vanilla_Decoder(nn.Module):
    def __init__(self):
        super(UNet_Encoder_Vanilla_Decoder, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoders = nn.ModuleList()
        for _ in range(6):
            self.encoders.append(UNet_Encoder(3,32))    # n_channels (input channels) = 3 & n_classes (output channels) = 32
        self.decoder = Mini_Decoder()

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        x = x = ((x-x.min())/(x.max()-x.min())) - 0.5
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoders[i](x[i]))
        encoder_output = torch.stack(encoder_outs).permute(0,2,1,3,4)
        encoder_output = torch.cat([i for i in encoder_output]).permute(1,0,2,3)
        decoder_output = self.decoder(encoder_output)
        return decoder_output


# ---------------- UNET ENCODER ---------------- #
class UNet_Encoder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet_Encoder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        #Input Size:- 3 x 256 x 308
        self.inc = DoubleConv(n_channels, 64)
        #Current Size:- 64 x 256 x 308 or 64 x 252 x 304 if unpadded
        self.down1 = Down(64, 128)
        #Current Size:- 128 x 128 x 154 or 128 x 122 x 148 if unpadded
        self.down2 = Down(128, 256)
        #Current Size:- 256 x 64 x 77 or 256 x 57 x 70 if unpadded
        self.down3 = Down(256, 512)
        #Current Size:- 512 x 32 x 38 or 512 x 24 x 31 if unpadded
        factor = 2 if bilinear else 1   # assuming bilinear=True
        self.down4 = Down(512, 1024 // factor)
        #Current Size:- 512 x 16 x 19 or 512 x 12 x 15 if unpadded
        self.up1 = Up(1024, 512 // factor, bilinear)
        #Current Size:- 256 x 32 x 38 if padded
        self.up2 = Up(512, 256 // factor, bilinear)
        #Current Size:- 128 x 64 x 76 if padded
        self.up3 = Up(256, 128 // factor, bilinear)
        #Current Size:- 64 x 128 x 152 if padded
        self.up4 = Up(128, 64, bilinear)
        #Current Size:- 32 x 256 x 304 if padded
        self.outc = OutConv(64, n_classes)
        #Current Size:- 64 x 128 x 152 if padded

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
        return logits


# ---------------- UNET DECODER ---------------- #
class UNet_Decoder(nn.Module):
    def __init__(self):
        super(UNet_Decoder, self).__init__()
#        #Size:- 192 x 128 x 152
        self.decoder_features = nn.Sequential(

            nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True),

#            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(192,192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192,192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(96,96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96,96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(96, 48, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(48,32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32,16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 1, kernel_size=3, padding=1),

            #Current Size:- 1 x 800 x 800
        )
        
    def forward(self,x):
        return self.decoder_features(x)


# ---------------- UNET PARTS ---------------- #
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



