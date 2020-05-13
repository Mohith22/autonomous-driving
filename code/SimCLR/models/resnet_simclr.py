import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import torch
import copy


class ResNetSimCLR(nn.Module):

    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
                            "resnet50": models.resnet50(pretrained=False)}

        resnet = self._get_basemodel(base_model)
        num_ftrs = resnet.fc.in_features

        self.features = nn.Sequential(*list(resnet.children())[:-1])

        # projection MLP
        self.l1 = nn.Linear(num_ftrs, num_ftrs)
        self.l2 = nn.Linear(num_ftrs, out_dim)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
            print("Feature extractor:", model_name)
            return model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, x):
        h = self.features(x)
        h = h.squeeze()

        x = self.l1(h)
        x = F.relu(x)
        x = self.l2(x)
        return h, x




# ---------------- UNET ENCODER AND DECODER SimCLR ---------------- #
class UNet_Encoder_Decoder_SimCLR(nn.Module):
    def __init__(self, in_channels=3, args=None):
        super(UNet_Encoder_Decoder_SimCLR, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoders = nn.ModuleList()
        for _ in range(1):
            self.encoders.append(UNet_Encoder(in_channels,32))    # n_channels (input channels) = 3 & n_classes (output channels) = 32
        self.decoder = UNet_Decoder_SimCLR()
        self.sigmoid = nn.Sigmoid()
        self.depth_encoders = nn.ModuleList()
        self.siamese = args.siamese
        self.depth_avail = args.depth_avail
        self.use_orient_net = args.use_orient_net
        if (self.use_orient_net):
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
            for _ in range(1):
                self.depth_encoders.append(UNet_Encoder(1,32))
        elif (args.siamese):
            self.siamese_encoder = UNet_Encoder(in_channels,32)
            self.siamese_depth_encoder = UNet_Encoder(1,32)

    def forward(self, x):
        if (self.siamese):
            encoder_outs = []
            for i in range(1):
                print('shape of x is: ',x.shape)
                encoder_outs.append(self.siamese_encoder(x[:,i,:,:,:][:,0:3,:,:]))

            if (self.depth_avail):
                for i in range(1):
                    depth_encoder_out = self.siamese_depth_encoder(x[:,i,:,:,:][:,3,:,:].unsqueeze(dim=1))
                    encoder_outs[i] = torch.cat((encoder_outs[i],depth_encoder_out), dim=1)

        else:
            encoder_outs = []
            for i in range(1):
                encoder_outs.append(self.encoders[i](x[:,i,:,:,:][:,0:3,:,:]))

            if (self.depth_avail):
                for i in range(1):
                    depth_encoder_out = self.depth_encoders[i](x[:,i,:,:,:][:,3,:,:].unsqueeze(dim=1))
                    encoder_outs[i] = torch.cat((encoder_outs[i],depth_encoder_out), dim=1)

        if (self.use_orient_net):
            for i in range(1):
                encoder_outs[i] = self.orient_net(encoder_outs[i])
            
        encoder_output = torch.cat(encoder_outs,dim=1)
        if (not self.use_orient_net):
            encoder_output= self.enc_bottleneck(encoder_output)
        decoder_output = self.decoder(encoder_output)
        return encoder_output, decoder_output





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








# ---------------- UNET DECODER SimCLR---------------- #
class UNet_Decoder_SimCLR(nn.Module):
    def __init__(self):
        super(UNet_Decoder_SimCLR, self).__init__()
#        #Size:- 192 x 128 x 152
        self.decoder_features = nn.Sequential(

            nn.Upsample(size=(200, 200), mode='bilinear', align_corners=True),

#            nn.ConvTranspose2d(384, 192, kernel_size=4, stride=2, padding=1),

#            nn.Conv2d(192,192, kernel_size=3, padding=1),
#            nn.BatchNorm2d(192),
#            nn.ReLU(inplace=True),

#            nn.Conv2d(192,192, kernel_size=3, padding=1),
#            nn.BatchNorm2d(192),
#            nn.ReLU(inplace=True),

#            nn.Conv2d(192, 192, kernel_size=3, padding=1),
#            nn.BatchNorm2d(192),
#            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(192, 96, kernel_size=4, stride=2, padding=1),

#            nn.Conv2d(96,96, kernel_size=3, padding=1),
#            nn.BatchNorm2d(96),
#            nn.ReLU(inplace=True),

#            nn.Conv2d(96,96, kernel_size=3, padding=1),
#            nn.BatchNorm2d(96),
#            nn.ReLU(inplace=True),

#            nn.Conv2d(96, 96, kernel_size=3, padding=1),
#            nn.BatchNorm2d(96),
#            nn.ReLU(inplace=True),

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

        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 200)
        
    def forward(self,x):
        x = self.decoder_features(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 400)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x







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




