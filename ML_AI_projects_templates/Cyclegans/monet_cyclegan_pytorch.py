# %% [code]
# Dependencies
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import PIL
import os
import torch

from kaggle_datasets import KaggleDatasets


# %% [code]

# Create a set for monet labels and monet paintings
monet_set = []
monet_labels = []
# Iterate through monet jpg folder and use PIL to extract the image. Append the label and the image
# to their respective arrays
for dirname, _, filenames in os.walk('/kaggle/input/gan-getting-started/monet_jpg'):
    for filename in filenames:
        monet_set.append(PIL.Image.open(os.path.join(dirname, filename)))
        monet_labels.append(1)

# Create a 4D stack of images (#images, 256 width, 256 height, 3 channels)
monet_set = np.stack(monet_set)
monet_set = np.float32(monet_set)
monet_labels = np.array(monet_labels)
print(monet_set.shape)
print(monet_labels.shape)
# Print shapes for verification on notebook

# Create a set for photo labels and photos
photo_set = []
photo_labels = []
# Iterate through monet jpg folder and use PIL to extract the image. Append the label and the image
# to their respective arrays
i = 0
for dirname, _, filenames in os.walk('/kaggle/input/gan-getting-started/photo_jpg'):
    for filename in filenames:
        photo_set.append(PIL.Image.open(os.path.join(dirname, filename)))
        photo_labels.append(0)
        i = i + 1
        if i > 3000:
            break

# Create a 4D stack of images (#images, 256 width, 256 height, 3 channels)
photo_set = np.stack(photo_set)
photo_set = photo_set[0:3000]
photo_set = np.float32(photo_set)
photo_labels = np.array(photo_labels)
photo_labels = photo_labels[0:300]
print(photo_set.shape)
print(photo_labels.shape)
# Print shapes for verification on notebook


# %% [code]
# Alternative using numpy
# Plot figures to verify that images are correctly oriented.
# Scale images from 0->255 to 0->1
photo_set = photo_set / 255.
monet_set = monet_set / 255.
plt.subplot(121)
plt.title('Photo')
plt.imshow(photo_set[0])

plt.subplot(122)
plt.title('Monet')
plt.imshow(monet_set[0])

# %% [code]
class downsample(torch.nn.Module):
    def __init__(self, infilters, outfilters, kernel_size, dropout = 0):
        super(downsample, self).__init__()
        self.conv1 = torch.nn.Conv2d(infilters, outfilters, kernel_size, stride = 2, padding = 1)
        self.leaky = torch.nn.LeakyReLU(0.1)
        self.instancenorm = torch.nn.InstanceNorm2d(256)
        self.dropout = torch.nn.Dropout2d(p=dropout)

    def forward(self, x , apply_instancenorm = True):
        x = self.conv1(x)
        if apply_instancenorm:
            x = self.instancenorm(x)
        x = self.dropout(x)
        x = self.leaky(x)
        return x

class upsample(torch.nn.Module):
    def __init__(self, infilters, outfilters, kernel_size, dropout = 0):
        super(upsample, self).__init__()
        self.convT1 = torch.nn.ConvTranspose2d(infilters, outfilters, kernel_size, stride = 2, padding = 1)
        self.leaky = torch.nn.LeakyReLU(0.1)
        self.instancenorm = torch.nn.InstanceNorm2d(outfilters)
        self.dropout = torch.nn.Dropout2d(p=dropout)

    def forward(self, x , apply_instancenorm = True):
        x = self.convT1(x)
        if apply_instancenorm:
            x = self.instancenorm(x)
        x = self.dropout(x)
        x = self.leaky(x)
        return x


# %% [code]
class Gen(torch.nn.Module):
    def __init__(self):
        super(Gen, self).__init__()
        self.downstack = [
            downsample(3,64,4).cuda(), #128
            downsample(64,128,4).cuda(), #64
            downsample(128,256,4).cuda(), #32
            downsample(256,512,4).cuda(), #16
            downsample(512,512,4).cuda(), #8
            downsample(512,512,4).cuda(), #4
            downsample(512,512,4).cuda(), #2
            downsample(512,512,4).cuda(), #1
        ]
        self.upstack = [
            upsample(512, 512, 4, dropout = 0.2).cuda(), #2
            upsample(1024, 512, 4, dropout = 0.2).cuda(), #4
            upsample(1024, 512, 4, dropout = 0.2).cuda(), #8
            upsample(1024, 512, 4).cuda(), #16
            upsample(1024, 256, 4).cuda(), #32
            upsample(512, 128, 4).cuda(), #64
            upsample(256, 64, 4).cuda(), #128
        ]
        self.last = torch.nn.ConvTranspose2d(128, 3, 4, stride = 2, padding = 1)
    def forward(self,x):
        resid_x = []
        i = 0
        for op in self.downstack:
            
            if x.shape[2] == 2:
                x = op(x, apply_instancenorm=False)
            else:
                x = op(x)
            resid_x.append(x)
            i = i + 1
            
        for op, res in zip(self.upstack, reversed(resid_x[:-1])):
            x = op(x)
            x = torch.cat((x,res), dim = 1)
          
        x = self.last(x)
        x = torch.tanh(x)
        return(x)


# %% [code]
class Disc(torch.nn.Module):
    def __init__(self):
        super(Disc, self).__init__()
        self.conv1 = downsample(3,64,4)
        self.conv2 = downsample(64,128,4)
        self.conv3 = downsample(128,256,4)
        self.instancenorm = torch.nn.InstanceNorm2d(256)
        self.last = torch.nn.Conv2d(256, 1, 4, stride = 1)
        self.leak = torch.nn.LeakyReLU(0.1)
    def forward(self, x):
        x = self.conv1(x)
        
        x = self.conv2(x)
        
        x = self.conv3(x)
        
        x = self.instancenorm(x)
        x = self.leak(x)
        x = self.last(x)
        
        return x

# %% [code]
def disc_loss(real, gen):
    crit = torch.nn.BCEWithLogitsLoss()
    l1 = crit(real, torch.zeros_like(real))
    l2 = crit(gen, torch.ones_like(gen))
    return_loss = (l1+l2)/2.
    return (return_loss)
def gen_loss(gen):
    crit = torch.nn.BCEWithLogitsLoss()
    return crit(gen, torch.ones_like(gen))
    
def calc_cycle_loss(R_image, C_image, LAMBDA):
    # get the mean of the difference between real and cycled image
    loss = torch.mean(torch.abs(R_image - C_image))
    return LAMBDA * loss
def identity_loss(R_image, I_image, LAMBDA):
    # get the mean of the difference between the an image and the image going through
    # the generator that produces the same type of image
    loss = torch.mean(torch.abs(R_image - I_image))
    return LAMBDA * 0.5 * loss
class Pix2Pix():
    def __init__(self):
        
        self.monet_gen = Gen()
        self.monet_gen.cuda()
        self.photo_gen = Gen()
        self.photo_gen.cuda()
        self.monet_disc = Disc()
        self.monet_disc.cuda()
        self.photo_disc = Disc()
        self.photo_disc.cuda()
        self.lambda_cycle = 10
    def compile(
        # Create the optimizers for the 
        # 4 networks and a loss function
        # for the generator, discriminator,
        # the cycle, and identity.
        self,
        m_gen_opt,
        p_gen_opt,
        m_disc_opt,
        p_disc_opt,
        gen_loss_fn,
        disc_loss_fn,
        cycle_loss_fn,
        identity_loss_fn
    ):
        # Create the optimizers for the 
        # 4 networks and a loss function
        # for the generator, discriminator,
        # the cycle, and identity.
        
        self.m_gen_opt = m_gen_opt
        self.p_gen_opt = p_gen_opt
        self.m_disc_opt = m_disc_opt
        self.p_disc_opt = p_disc_opt
        self.gen_loss_fn = gen_loss_fn
        self.disc_loss_fn = disc_loss_fn
        self.cycle_loss_fn = cycle_loss_fn
        self.identity_loss_fn = identity_loss_fn


    def train(self, data):
        
        R_monet, R_photo = data
        
        F_monet = self.monet_gen(R_photo)
        F_photo = self.photo_gen(R_monet)
        
        I_monet = self.monet_gen(R_monet)
        I_photo = self.photo_gen(R_photo)
        
        C_monet = self.monet_gen(F_photo)
        C_photo = self.photo_gen(F_monet)
        
        disc_R_monet = self.monet_disc(R_monet)
        disc_R_photo = self.photo_disc(R_photo)
        
        disc_F_monet = self.monet_disc(F_monet)
        disc_F_photo = self.photo_disc(F_photo)
        
        
        monet_disc_loss = self.disc_loss_fn(disc_R_monet.detach(), disc_F_monet.detach())
        photo_disc_loss = self.disc_loss_fn(disc_R_photo.detach(), disc_F_photo.detach())
        
        monet_gen_loss = self.gen_loss_fn(disc_F_monet)
        photo_gen_loss = self.gen_loss_fn(disc_F_photo)
        
        
        total_cycle_loss = self.cycle_loss_fn(R_monet, C_monet, self.lambda_cycle) + self.cycle_loss_fn(R_photo, C_photo, self.lambda_cycle)
        
        total_monet_gen_loss = monet_gen_loss + total_cycle_loss + self.identity_loss_fn(R_monet, I_monet, self.lambda_cycle)
        total_photo_gen_loss = photo_gen_loss + total_cycle_loss + self.identity_loss_fn(R_photo, I_photo, self.lambda_cycle)
        
        
        self.m_gen_opt.zero_grad()
        self.p_gen_opt.zero_grad()
        self.m_disc_opt.zero_grad()
        self.p_disc_opt.zero_grad()
        (total_monet_gen_loss + total_photo_gen_loss + monet_disc_loss + photo_disc_loss).backward()
        
        '''
        total_monet_gen_loss.backward()
        total_photo_gen_loss.backward()
        monet_disc_loss.backward()
        photo_disc_loss.backward()
        '''
        self.p_gen_opt.step()
        self.p_disc_opt.step()
        self.m_gen_opt.step()
        self.m_disc_opt.step()
        
        
        loss_l = {
            "monet_gen_loss": total_monet_gen_loss,
            "photo_gen_loss": total_photo_gen_loss,
            "monet_disc_loss": monet_disc_loss,
            "photo_disc_loss": photo_disc_loss,
            "cycle": total_cycle_loss
        }
        
        return loss_l
        
        
        
    
        
cloned_photo = photo_set.copy()
cloned_photo = photo_set
cloned_photo = np.transpose(cloned_photo, (0,3,2,1))
cloned_photo = torch.Tensor(cloned_photo).cuda()

cloned_monet = monet_set.copy()
cloned_monet = monet_set
cloned_monet = np.transpose(cloned_monet, (0,3,2,1))
cloned_monet = torch.Tensor(cloned_monet).cuda()

mod = Pix2Pix()
monet_gen_opt =  torch.optim.Adam(mod.monet_gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
photo_gen_opt =  torch.optim.Adam(mod.photo_gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
monet_disc_opt =  torch.optim.Adam(mod.monet_disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
photo_disc_opt =  torch.optim.Adam(mod.photo_disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

mod.compile(
        m_gen_opt = monet_gen_opt,
        p_gen_opt = photo_gen_opt,
        m_disc_opt = monet_disc_opt,
        p_disc_opt = photo_disc_opt,
        gen_loss_fn = gen_loss,
        disc_loss_fn = disc_loss,
        cycle_loss_fn = calc_cycle_loss,
        identity_loss_fn = identity_loss
    )

for e in range(0,5):
    for i in range(0,10000):
        p1 = np.random.randint(0,300)
        p2 = np.random.randint(0,3000)
        cmonet = cloned_monet[p1:p1+1]
        cphoto = cloned_photo[p2:p2+1]
        loss_l = mod.train([cphoto, cmonet])
        if i%25 == 0:
            print(i)
            for key in loss_l:
                print(key, ' : ', loss_l[key])
