from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from models import *
import numpy as np
import matplotlib.pyplot as plt
from my_utils import gradient_magnitude



# parse augments ==========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  help='cifar10 | lsun | imagenet | folder | lfw | fake' ,default='lfw')# required=True
parser.add_argument('--dataroot', help='path to dataset',default='./../data/')# required=True
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=2500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda' ,default= True)
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./saves', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')


opt = parser.parse_args()
print(opt)



opt.manualSeed = 4949
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

opt.dataroot = os.path.join(opt.dataroot,opt.dataset)
opt.outf = os.path.join(opt.outf,opt.dataset, str(opt.manualSeed) + 'grad',)


cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(root=opt.dataroot,  classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Resize(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Resize(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers),drop_last=True)

device = torch.device("cuda:0" if opt.cuda else "cpu")
ngpu = int(opt.ngpu)

nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3
imagesize = opt.imageSize
#load checkpoint
#opt.netG = ''
#opt.netD = ''
#==========================================================================================



#====== model and optimizer =========
netG = Generator(ngpu,nz = nz, ngf = ngf, nc = nc , imagesize = imagesize ).to(device)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = Discriminator(ngpu,nc = nc , ndf= ndf, imagesize = imagesize).to(device)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)


fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
real_label = 1
fake_label = 0

criterion = nn.BCELoss()

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


def lossFuntion(output, label):
    #== -1*(  label * log(output) + (1-label) * log(1- output)  )
    #== my version : -1* (  label * log(output) -  (1-label) * log(output)  )
    loss = -1 * ( label * torch.log(output) + (1-label)* torch.log(1-output))
    loss = torch.mean(loss)
    return loss

def lossMSE(output, label):
    loss = torch.pow((label-output),2)
    return torch.mean(loss)
def myLoss (output, label):
    return torch.nn.MSELoss()(output,label) + torch.nn.BCELoss()(output,label)





def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    #plt.imshow(img)
    print(img.size())
    plt.imshow(np.transpose(npimg, (1, 2,0)),cmap='gray')
    #plt.imshow(npimg,cmap='gray')
    plt.show()



#====== train and validate =========
for epoch in range(opt.niter):
    for i, (real_data,_) in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = real_data
        real_data = real_data.to(device)
        real_g = gradient_magnitude(real_data,3)

        label = torch.ones((opt.batchSize,)).to(device)
        output = netD(real_data)
        output_g = netD(real_g)
        errD_real = criterion(output, label)
        errD_real.backward()
        errD_real_g = criterion(output_g,label)
        errD_real_g.backward()
        D_x = output.mean().item()

        # train with fake
        noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)
        fake = netG(noise)
        fake_g = gradient_magnitude(netG(noise),3)

        label.fill_(fake_label)
        output = netD(fake.detach())
        output_g = netD(fake_g.detach())
        errD_fake = criterion(output, label)
        errD_fake.backward()
        errD_fake_g = criterion(output_g, label)
        errD_fake_g.backward()
        D_G_z1 = output.mean().item()

        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        fake = netG(noise)
        fake_g = gradient_magnitude(netG(noise),3)
        output = netD(fake)
        output_g = netD(fake_g)
        '''
        errG = criterion(output, label)
        errG.backward()
        errG_g = criterion(output_g, label)
        errG_g.backward()
        '''

        #'''
        loss1 = -1 * criterion(output, torch.full((opt.batchSize,), 0, device=device))
        loss1_g = -1 * criterion(output_g, torch.full((opt.batchSize,), 0, device=device))

        loss2 = criterion(output,torch.full((opt.batchSize,), 1, device=device))
        loss2_g = criterion(output_g, torch.full((opt.batchSize,), 1, device=device))
        if (torch.abs(loss1) < torch.abs(loss2)):
            errG = loss2 +loss2_g
            print(round(loss1.item(), 4), round(loss2.item(), 4))
        else:
            errG = loss1 +loss1_g
            print(round(loss1.item(), 4), round(loss2.item(), 4))
        # loss = criterion(output_fake,label.fill_(0))
        errG.backward()
        #'''


        D_G_z2 = output.mean().item()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            if not os.path.isdir(os.path.join(opt.outf)):
                os.makedirs(os.path.join(opt.outf), exist_ok=True)
            vutils.save_image(real_cpu,'%s/real_samples.png' % opt.outf, normalize=True)


            fake = netG(fixed_noise)
            if not os.path.isdir(os.path.join(opt.outf)):
                os.makedirs(os.path.join(opt.outf), exist_ok=True)
            vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)


    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
