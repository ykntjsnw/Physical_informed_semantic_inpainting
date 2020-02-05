#!/usr/bin/env python
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
#from tensorboardX import SummaryWriter
from WGANGP_with_PI import Generator, Discriminator, calc_horizontal_grad, calc_vertical_grad

import os
import argparse


# In[ ]:


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nc", type=int, default=4)
    parser.add_argument("--nz", type=int, default=100)    

    parser.add_argument("--ngf", type=int, default=64)
    parser.add_argument("--ndf", type=int, default=64)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--state-dict-D", type=str, default="params/netD.pkl")
    parser.add_argument("--state-dict-G", type=str, default="params/netG.pkl")
    parser.add_argument("--optim-steps", type=int, default=1500)
    parser.add_argument("--blending-steps", type=int, default=3000)
    parser.add_argument("--BATCH-SIZE", type=int, default=64)
    parser.add_argument("--LAMBDA-P", type=float, default=0.1, help="p_loss penalty lambda hyperparameter")
    parser.add_argument("--norm", type=str, choices=["L1", "L2"], default="L1", help="norm type used in context loss, L1 or L2")
    parser.add_argumenta("--weighted-mask", action='store_true', default=False)
    parser.add_argument("--window-size", type=int, default=25)
    
    args = parser.parse_args()
    return args


def context_loss(norm, corrupted_images, generated_images, masks):
    if norm == "L1":
        return torch.sum((torch.abs((corrupted_images - generated_images) * masks))
    else:
        assert norm == "L2"
        return torch.sum(((corrupted_images - generated_images) ** 2) * masks))


def posisson_blending(masks, generated_images, corrupted_images):
    
    print("Starting Poisson blending ...")
    
    initial_guess = masks * corrupted_images + (1 - masks) * generated_images
    image_optimum = nn.Parameter(torch.FloatTensor(initial_guess.detach().cpu().numpy()).to(device))
    optimizer_blending = optim.Adam([image_optimum])
    generated_grad_h = calc_horizontal_grad(generated_images)
    generated_grad_v = calc_vertical_grad(generated_images)

    for epoch in range(args.blending_steps):
        optimizer_blending.zero_grad()
        image_optimum_grad_h = calc_horizontal_grad(image_optimum)
        image_optimum_grad_v = calc_vertical_grad(image_optimum)
        blending_loss = torch.sum(((generated_grad_h - image_optimum_grad_h) ** 2 + (generated_grad_v - image_optimum_grad_v) ** 2) * (1 - masks))
        blending_loss.backward()
        image_optimum.grad = image_optimum.grad * (1 - masks)
        optimizer_blending.step()

        if epoch % 100 ==99:
            print("Epoch: {}/{} \t Blending loss: {:.3f}".format(epoch, args.blending_steps, blending_loss)) 

    del optimizer_blending
    return image_optimum.detach()    



def get_weighted_mask(mask, window_size=25):
    assert len(mask.shape) == 3
    assert window_size % 2 == 1
    max_shift = window_size // 2
    windows = torch.ones((max_shift, max_shift)).reshape(1, 1, max_shift, max_shift)
    windows = windows.to(device)
    weighted_mask = F.conv2d(mask.unsqueeze(1), windows, padding=max_shift)
    output = 1 - output / (window_size ** 2 - 1)
    output = output.squeeze(1)
    return output * mask
                                 
                         
def inpaint(args):
    
    # Decide which device we want to run on
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    
    ###dataloader = 待完成。。。
    
    # load the model parameters after GAN training
    netG = Generator(args).to(device)
    netG.load_state_dict(torch.load(args.state_dict_G))
    netD = Discriminator(args).to(device)
    netD.load_state_dict(torch.load(args.state_dict_D))
    
    #netD.train()
    #netG.train()
    
    for i, (corrupted_images, original_images, masks) in enumerate(dataloader):
        corrupted_images, masks = corrupted_images.to(device), masks.to(device)
        if args.weighted_mask == True:
            masks = get_weighted_mask(masks, args.window_size)
        z_optimum = nn.Parameter(torch.FloatTensor(np.random.normal(0, 1, (corrupted_images.size(0),args.nz))).to(device))
        optimizer_inpaint = optim.Adam([z_optimum])       
        
        print("Starting backprop to input ...")
        for epoch in range(args.optim_steps):
            optimizer_inpaint.zero_grad()
            generated_images = netG(z_optimum)
            D_fake = netD(fake_data)
            c_loss = context_loss(args.norm, corrupted_images, generated_images, masks)
            p_loss = torch.sum(- netD(generated_images))
            inpaint_loss = c_loss + args.LAMBDA_P * p_loss
            inpaint_loss.backward()
            optimizer_inpaint.step()
            if epoch % 100 == 99:
                print("Epoch: {}/{} \tContext_loss: {:.3f} \tPrior_loss: {:.3f} \tInpaint_loss: {:.3f}]".format(epoch, args.optim_steps, c_loss, p_loss, inpaint_loss))
            
        blended_images = posisson_blending(masks, generated_images.detach(), corrupted_images)
    
        image_range = torch.min(corrupted_images), torch.max(corrupted_images)
        
        save_image(corrupted_images, "../outputs/corrupted_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(generated_images, "../outputs/output_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(blended_images, "../outputs/blended_{}.png".format(i), normalize=True, range=image_range, nrow=5)
        save_image(original_images, "../outputs/original_{}.png".format(i), normalize=True, range=image_range, nrow=5)

        del z_optimum, optimizer_inpaint

        
if __name__ == "__main__":
    args = get_arguments()
    inpaint(args)

