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
from my_utils import *
import matplotlib.pyplot as plt


def train_and_validation(opt,dataloader):


    #=== parse opt =============
    device = torch.device("cuda:0" if opt.cuda else "cpu")
    ngpu = int(opt.ngpu)

    nz = int(opt.nz)
    ngf = int(opt.ngf)
    ndf = int(opt.ndf)
    nc = 3 #num channel
    imagesize = opt.imageSize
    #load checkpoint by inline code
    #opt.netG = './saves/lfw/netG_epoch_12.pth'
    #opt.netD = ''
    #============================



    #====== model and optimizer =========
    netG = Generator(ngpu,nz = nz, ngf = ngf, nc = nc , imagesize = imagesize).to(device)
    netG.apply(weights_init)
    if opt.netG != '':
        netG.load_state_dict(torch.load(opt.netG))
    print(netG)

    netD = Discriminator(ngpu,nc = nc , ndf= ndf,imagesize = imagesize).to(device)
    netD.apply(weights_init)
    if opt.netD != '':
        netD.load_state_dict(torch.load(opt.netD))
    print(netD)

    criterion = nn.BCELoss()
    optimizer_netD = torch.optim.Adam(netD.parameters(),lr = opt.lr,betas = (opt.beta1,0.9999))
    optimizer_netG = torch.optim.Adam(netG.parameters(),lr = opt.lr, betas = (opt.beta1,0.9999))

    fixed_noise = torch.randn(opt.batchSize, nz, 1, 1, device=device)

    for epo in range(opt.nepo):
        for itr, (data,label) in enumerate(dataloader):
            #=================================
            # (1) update Discriminator :
            #  maximize log D(x) + log (1 - D(G(z))
            #  = minimize -1* (log D(x) + log( 1- D(G(z)))
            #=================================
            optimizer_netD.zero_grad()
            real_data_cpu = data
            real_data = data.to(device)

            label = torch.full((opt.batchSize,), 1, device=device)
            output = netD(real_data)
            loss_Real = criterion(output,label)
            loss_Real.backward()

            label = label.fill_(0)
            noise = torch.randn((opt.batchSize,nz,1,1),device = device)
            fake = netG(noise)
            output_fake = netD(fake.detach())
            loss_Fake = criterion(output_fake,label)
            loss_Fake.backward()
            optimizer_netD.step()

            loss_D = (loss_Real + loss_Fake).item()
            # =================================
            # (1) update Generator :
            #  maximize log (D(G(z))
            #  = minimize -1* (log( D(G(z)))
            # =================================
            optimizer_netG.zero_grad()
            noise = torch.randn((opt.batchSize,nz,1,1,), device = device)
            fake = netG(noise)
            output_fake = netD(fake)
            label = label.fill_(1)


            loss1 = -1 * criterion(output_fake, torch.full((opt.batchSize,), 0, device=device))
            loss2 = criterion(output_fake, label.fill_(1))
            if(torch.abs(loss1) < torch.abs(loss2)):
                loss = loss2
                print(round(loss1.item(),4), round(loss2.item(),4))
            else:
                loss = loss1
                print(round(loss1.item(), 4), round(loss2.item(), 4))
            #loss = criterion(output_fake,label.fill_(0))
            loss.backward()
            optimizer_netG.step()

            loss_G = loss.item()
            print("itr:",itr ,"loss_D : ", round(loss_D,4), "loss_G : ", round(loss_G,4) , "D(fake) :", round(torch.mean(output_fake).detach().item(),4))

            if itr% 50  == 0:
                ''' for display
                imshow_with_tensor(vutils.make_grid(fake.detach()).cpu())
                if not os.path.isdir(os.path.join(opt.outf)):
                    os.makedirs(os.path.join(opt.outf), exist_ok=True)
                vutils.save_image(fake.detach(),'%s/fake_samples_epoch_%03d.png' % (opt.outf, epo))
                '''
                if not os.path.isdir(os.path.join(opt.outf)):
                    os.makedirs(os.path.join(opt.outf), exist_ok=True)
                vutils.save_image(real_data_cpu, '%s/real_samples.png' % opt.outf, normalize=True)

                fake = netG(fixed_noise)
                if not os.path.isdir(os.path.join(opt.outf)):
                    os.makedirs(os.path.join(opt.outf), exist_ok=True)
                vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (opt.outf, epo),
                                  normalize=True)


                # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epo))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epo))
