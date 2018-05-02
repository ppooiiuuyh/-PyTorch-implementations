import torch.nn as nn
import torch
import math
# shared global variables
# called firstly on here (by import at main.py)
# then initialized with actual value at main.py
# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self, ngpu,nz,ngf,nc,imagesize):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        log_imagesize = int(math.log(imagesize,2))
        sequence = []
        # input layer
        sequence += [
            nn.ConvTranspose2d(nz, ngf * 2**(log_imagesize-3), 4, 2, 0, bias=False),
            nn.BatchNorm2d(ngf * 2**(log_imagesize-3)),
            nn.ReLU(True)
        ]
        for i in range( log_imagesize-3 ):
            #print(i ,log_imagesize-3)
            sequence += [
                nn.ConvTranspose2d(ngf * 2**(log_imagesize-3-i), ngf * 2**(log_imagesize-3-i-1), 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2**(log_imagesize-3-i-1)),
                nn.ReLU(True)
            ]

        #output layer
        sequence += [
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            #nn.ReLU()
        ]
        self.main = nn.Sequential(*sequence)


    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output



class Discriminator(nn.Module):
    def __init__(self, ngpu,ndf,nc,imagesize):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True)
        )

            # input is (nc) x 64 x 64

        log_imagesize = int(math.log(imagesize,2))
        sequence = []
        for i in range(log_imagesize-3):
            sequence += [
                nn.Conv2d(ndf*2**i, ndf * 2**(i+1), 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2**(i+1)),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        self.hidden = nn.Sequential(*sequence)

        self.output =  nn.Sequential(
            nn.Conv2d(ndf * 2**(log_imagesize-3), 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            #nn.Tanh()
        )
        '''
        self.hidden = nn.Sequential(
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        '''

    def forward(self, input):
        if input.is_cuda and self.ngpu > 1:
            output = self.conv1(input)
            output = nn.parallel.data_parallel(self.main, output, range(self.ngpu))
        else:
            output = self.conv1(input)
            output = self.hidden(output)
            output = self.output(output)
            #print(output)
        return output.view(-1, 1).squeeze(1)

