import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
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
            #Current Size:- 64s x 64 x 76
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            #Current Size:- 64 x 64 x 76
        )

    def forward(self, x):
        return self.encoder_features(x)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        #Input Size:- 64 x 64 x 76
        self.decoder_features = nn.Sequential(
        	nn.Upsample(size=(100,100), mode='bilinear', align_corners=True)
        	#Current Size:- 64 x 100 x 100
        	nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1)
        	#Current Size:- 32 x 200 x 200
        	nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1)
        	#Current Size:- 16 x 400 x 400
        	nn.ConvTranspose2d(16, 1,kernel_size=3, stride=2, padding=1)
        	#Current Size:- 1 x 800 x 800
        )

    def forward(self, x):
        return self.decoder_features(x)


