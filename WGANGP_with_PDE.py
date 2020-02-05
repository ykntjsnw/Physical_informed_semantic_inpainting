#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tensorboardX import SummaryWriter
import os
import argparse


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument("--nz", type=int, default=100)    

    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--num-epochs", type=int, default=15000)
    parser.add_argument("--BATCH-SIZE", type=int, default=64)
    parser.add_argument("--LAMBDA", type=int, default=10, help="Gradient penalty lambda hyperparameter")
    parser.add_argument("--lr", type=float, default=0.0002, help="Learning rate for optimizers")
    parser.add_argument("--beta1", type=float, default=0.5, help="Beta1 hyperparam for Adam optimizers")
    args = parser.parse_args()
    return args


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

        
class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(args.nz, args.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(args.ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(args.ngf * 8, args.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(args.ngf * 4, args.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(args.ngf * 2, args.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(args.ngf, args.nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)



class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.ngpu = args.ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(args.nc, args.ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(args.ndf, args.ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(args.ndf * 2, args.ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(args.ndf * 4, args.ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(args.ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(args.ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            # state size. 1 x 1 x 1 
        )

    def forward(self, input):
        return self.main(input).view(-1)
    
    



def calc_gradient_penalty(netD, real_data, fake_data):
    # print "real_data: ", real_data.size(), fake_data.size()
    BATCH_SIZE = real_data.size(0)
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, real_data.nelement()/BATCH_SIZE).contiguous().view(BATCH_SIZE, 4, 64, 64)
    alpha = alpha.to(device) 

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    
    interpolates = interpolates.to(device)
    interpolates.requires_grad_()

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    #gradients has the same size as interpolates
    gradients = gradients.view(gradients.size(0), -1)

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() 
    return gradient_penalty


def calc_horizontal_grad(im):
    sobel_filter = torch.FloatTensor([[1, 0, -1],[2, 0, -2],[1, 0, -1]]).reshape(1,1,3,3)
    sobel_filter = sobel_filter.to(device)
    horizontal_grad = F.conv2d(im, sobel_filter, padding=1) 
    return horizontal_grad


def calc_vertical_grad(im):
    sobel_filter = torch.FloatTensor([[1, 2, 1],[0, 0, 0],[-1, -2, -1]]).reshape(1,1,3,3)
    sobel_filter = sobel_filter.to(device)
    vertical_grad = F.conv2d(im, sobel_filter, padding=1) 
    return vertical_grad


if __name__ == "__main__":
    
    args = get_arguments()
    
    torch.manual_seed(1)
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.ngpu > 0) else "cpu")
    
    # Create the generator
    netG = Generator(args).to(device)

    # Handle multi-gpu if desired
    #if (device.type == 'cuda') and (ngpu > 1):
    #    netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netG.apply(weights_init)
    # Create the Discriminator
    
    netD = Discriminator(args).to(device)

    # Handle multi-gpu if desired
    #if (device.type == 'cuda') and (ngpu > 1):
    #    netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)


    writer = SummaryWriter(comment='WGANGP_with_PI')
    writer.add_graph(netD)
    writer.add_graph(netG)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    # create the dataloader
    # train_dataloader, dev_dataloader = 待完成...

    
    ### Training Loop

    for epoch in range(args.num_epochs):

        netD.train()
        netG.train()

        critic_train_loss = []
        Wasserstein_train =[]


        ## update D network
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in netG.parameters():  
            p.requires_grad = False  # freeze netG

        for i,real_data in enumerate(train_dataloader):

            netD.zero_grad()

            #train with real
            real_data = real_data.to(device)
            D_real = netD(real_data).mean() 


            #train with fake
            noise = torch.randn((args.BATCH_SIZE, args.nz))
            noise = noise.to(device)
            fake_data = netG(noise)
            D_fake = netD(fake_data).mean()


            #train with gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, real_data, fake_data)

            D_loss = D_fake - D_real + gradient_penalty * args.LAMBDA
            _Wasserstein_train = D_real - D_fake
            Wasserstein_train.append(_Wasserstein_train.cpu().item())
            critic_train_loss.append(D_loss.cpu().item())
            D_loss.backward()
            optimizerD.step()

        writer.add_scalar("critic_train_loss", np.mean(critic_train_loss), global_step=epoch)
        writer.add_scalar("wasserstein_train", np.mean(Wasserstein_train), global_step=epoch)

        #print train loss and Wasserstein every 100 epochs
        if epoch % 100 == 99:
            print("epoch:{} critic_train_loss:{}".format(epoch, np.mean(critic_train_loss)))
            print("epoch:{} Wasserstein_train:{}".format(epoch, np.mean(Wasserstein_train)))

        ##update G network
        for p in netD.parameters():  
            p.requires_grad = False  # freeze netD
        for p in netG.parameters():  
            p.requires_grad = True  # unfreeze netG

        netG.zero_grad()

        #train with original generator loss
        noise = torch.randn((args.BATCH_SIZE, args.nz))
        noise = noise.to(device)
        fake_data = netG(noise)
        D_fake = netD(fake_data).mean() 

        ## train with PDE constraint

        # constitutive constraint
        grad_h = calc_horizontal_grad(fake_data[:,1:2,:,:])
        grad_v = calc_vertical_grad(fake_data[:,1:2,:,:])
        flux_h_pred = torch.exp(fake_datae[:,0:1,:,:]) * grad_h                    
        flux_v_pred = torch.exp(fake_datae[:,0:1,:,:]) * grad_v                        
        loss_1 = torch.mean((fake_data[:,2:3,:,:] + flux_h_pred) ** 2) + torch.mean((fake_data[:,3:4,:,:] + flux_v_pred) ** 2)

        # continuity constraint
        div_h = calc_horizontal_grad(fake_data[:,2:3,:,:])
        div_v = calc_vertical_grad(fake_data[:,3:4,:,:])
        loss_2 = torch.mean((div_h + div_v) ** 2)

        # boundary constraint
        left_bound, right_bound = fake_data[:,1,:,0], fake_data[:,1,:,-1]
        top_flux, down_flux = fake_datae[:,3,0,:], fake_data[:,3,-1,:]
        loss_dirchlet = torch.mean((left_bound - 1.) ** 2) + torch.mean(right_bound ** 2)
        loss_neumann = torch.mean(top_flux ** 2) + torch.mean(down_flux ** 2)
        loss_boundary = loss_dirchlet + loss_neumann

        G_loss = -D_fake + loss_1 +loss_2 + 10.0 * loss_boundary 
        G_loss.backward()
        optimizerG.step()


        # Calculate dev wasserstein every 100 iters
        if epoch % 100 == 99:

            netD.eval()
            netG.eval()

            Wasserstein_dev = []
            with torch.no_grad():
                for i,real_data in enumerate(dev_dataloader):
                    real_data = real_data.to(device)
                    D_real = netD(real_data).mean()
                    noise = torch.randn((args.BATCH_SIZE, args.nz))
                    noise = noise.to(device)
                    fake_data = netG(noise)
                    D_fake = netD(fake_data).mean()
                    _Wasserstein_dev = D_real - D_fake
                    Wasserstein_dev.append(_Wasserstein_train.cpu().item())
                print("epoch:{} Wasserstein_dev:{}".format(epoch, np.mean(Wasserstein_dev)))

            writer.add_scalar("wasserstein_dev", Wasserstein_dev, global_step=epoch)

    writer.close()

    #save model params after training
    torch.save(netD, "params/netD.pkl")
    torch.save(netG, "params/netG.pkl") 
    
    

  

        
      
    
          
        
        
          
          
          
          
          

