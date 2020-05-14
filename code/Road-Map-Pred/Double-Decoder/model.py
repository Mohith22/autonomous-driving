# --Imports -- #
import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import torchvision.models as models

# Unet encoder based double architecture

# ---------------- UNET ENCODER AND DECODER ---------------- #
class UNet_Encoder_Decoder(nn.Module):
    def __init__(self, in_channels=3, args=None):
        super(UNet_Encoder_Decoder, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoders = nn.ModuleList()
        for _ in range(6):
            self.encoders.append(UNet_Encoder(in_channels,32))    # n_channels (input channels) = 3 & n_classes (output channels) = 32
        self.decoder = UNet_Decoder()
        self.depth_decoder = UNet_Depth_Decoder()
        self.sigmoid = nn.Sigmoid()
        self.depth_encoders = nn.ModuleList()
        self.siamese = args.siamese
        self.depth_avail = args.depth_avail
        self.use_orient_net = args.use_orient_net
        self.orient_net = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
            )
        self.enc_bottleneck = nn.Conv2d(384,192, kernel_size=3, padding=1)
        if (not args.siamese):
            for _ in range(6):
                self.depth_encoders.append(UNet_Encoder(1,32))
        elif (args.siamese):
            self.siamese_encoder = UNet_Encoder(in_channels,32)
            self.siamese_depth_encoder = UNet_Encoder(1,32)

    def forward(self, x):
        if (self.siamese):
            encoder_outs = []
            for i in range(6):
                encoder_outs.append(self.siamese_encoder(x[:,i,:,:,:][:,0:3,:,:]))

            if (self.depth_avail):
                for i in range(6):
                    depth_encoder_out = self.siamese_depth_encoder(x[:,i,:,:,:][:,3,:,:].unsqueeze(dim=1))
                    encoder_outs[i] = torch.cat((encoder_outs[i],depth_encoder_out), dim=1)

        else:
            encoder_outs = []
            for i in range(6):
                encoder_outs.append(self.encoders[i](x[:,i,:,:,:][:,0:3,:,:]))

            if (self.depth_avail):
                for i in range(6):
                    depth_encoder_out = self.depth_encoders[i](x[:,i,:,:,:][:,3,:,:].unsqueeze(dim=1))
                    encoder_outs[i] = torch.cat((encoder_outs[i],depth_encoder_out), dim=1)

        if (self.use_orient_net):
            for i in range(6):
                encoder_outs[i] = self.orient_net(encoder_outs[i])
        
        decoder_depth_outs = []
        for i in range(6):
          decoder_depth_outs.append(self.depth_decoder(encoder_outs[i]))
        decoder_depth_output = torch.cat(decoder_depth_outs,dim=1)
        encoder_output = torch.cat(encoder_outs,dim=1)
        #if (not self.use_orient_net):
        #    encoder_output= self.enc_bottleneck(encoder_output)
        decoder_downstream_output = self.decoder(encoder_output)
        return decoder_downstream_output, decoder_depth_output

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
        #Current Size:- 64 x 256 x 304 if padded
        self.outc = OutConv(64, 32)
        #Current Size:- 32 x 256 x 304 if padded
        self.outc1 = OutConv1(32, n_classes)
        #Current Size:- 32 x 128 x 152 if padded

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
        logits = self.outc1(x)
        return logits

# ---------------- UNET DECODER ---------------- #
class UNet_Depth_Decoder(nn.Module):
    def __init__(self):
        super(UNet_Depth_Decoder, self).__init__()
#        #Size:- 32 x 128 x 152
        self.decoder_features = nn.Sequential(

            nn.Upsample(size=(128, 153), mode='bilinear', align_corners=True),

#            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(32,64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64,128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),

            nn.Conv2d(96,96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96,96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),

            nn.Conv2d(96, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),

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
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class OutConv1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv1, self).__init__()
        self.outconv1_encoder_features = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.outconv1_encoder_features(x)




