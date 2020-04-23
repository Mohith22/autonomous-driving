import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

class SEncoder_Decoder(nn.Module):
    def __init__(self):
        super(SEncoder_Decoder, self).__init__()
        #Input Size:- 3 x 256 x 306
        self.encoders = nn.ModuleList()
        for _ in range(6):
            self.encoders.append(SEncoder())
        self.decoder = SDecoder()

    def forward(self, x):
        x = x.permute(1,0,2,3,4)
        x = x-0.5
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoders[i](x[i]))
        encoder_output = torch.stack(encoder_outs).permute(0,2,1,3,4)
        encoder_output = torch.cat([i for i in encoder_output]).permute(1,0,2,3)
        decoder_output = self.decoder(encoder_output)
        return decoder_output

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
        x = x-0.5
        encoder_outs = []
        for i in range(6):
            encoder_outs.append(self.encoders[i](x[i]))
        top_enc = torch.cat((encoder_outs[0], encoder_outs[1], encoder_outs[2]), dim=3)
        top_dec = torch.cat((encoder_outs[3], encoder_outs[4], encoder_outs[5]), dim=3)
        encoder_output = torch.cat((top_enc, top_dec), dim=2)
        #print(encoder_output.size()) - #Size - torch.Size([2, 32, 256, 468])
        decoder_output = self.decoder(encoder_output)
        return decoder_output

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

class SEncoder(nn.Module):
    def __init__(self):
        super(SEncoder, self).__init__()
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

class SDecoder(nn.Module):
    def __init__(self):
        super(SDecoder, self).__init__()
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
            nn.ConvTranspose2d(8, 2,kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 2 x 800 x 800
        )
        
    def forward(self,x):
        return self.decoder_features(x)


class Encoder_Decoder(nn.Module):
    def __init__(self):
        super(Encoder_Decoder, self).__init__()
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
        	nn.ReLU(inplace=True),
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





