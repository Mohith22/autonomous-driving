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





