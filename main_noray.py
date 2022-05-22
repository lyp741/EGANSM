 
import argparse
from cgi import test
import os
import numpy as np
import math
from sympy import viete
import torch

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from DCGAN_model import Generator, Discriminator
from policy import Policy
import time
import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib
import shutil

import random
# import testFID
# matplotlib.use("TkAgg")
shutil.rmtree('log_dir', ignore_errors=True)


os.makedirs("fake_images", exist_ok=True)
os.makedirs("real_images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=32, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
parser.add_argument("--pop_size", type=int, default=10, help="population size")
parser.add_argument("--fifth", type=int, default=1, help="fifth")
opt = parser.parse_args()
print(opt)
cuda = True if torch.cuda.is_available() else False
device = torch.device("cuda:0" if cuda else "cpu")


pop_size= opt.pop_size
fifth = opt.fifth
num_workers = pop_size*fifth
classfier = pop_size #+ 1

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

# Loss function
#adversarial_loss = torch.nn.BCELoss()
adversarial_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
generator = [Generator(opt.latent_dim).apply(weights_init_normal) for i in range(pop_size)]
discriminator = Discriminator(classfier)

if cuda:
    for g in generator:
        g.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Initialize weights
#generator.apply(weights_init_normal)
# discriminator.apply(weights_init_normal)

# Configure data loader
from sklearn import datasets

X,y = datasets.make_circles(n_samples=20)

#draw
import matplotlib.pyplot as plt

X = X[y == 0]
centers = X
#make blobs aroung centers
X, y = datasets.make_blobs(n_samples=200, centers=centers, cluster_std=0.04)

os.makedirs("../../data/mnist", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(torch.tensor(X)),
    batch_size=opt.batch_size,
    shuffle=True,
    # num_workers=20,
)


# Optimizers
#optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

#Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
Tensor = torch.FloatTensor

# @ray.remote
class Worker(object):
    #def __init__(self, policy):
         #self.policy = policy

    def do_rollouts(self, z, parent_param, generator, discriminator, j):
        label_G = Variable(torch.LongTensor(imgs.shape[0]),requires_grad=False).to(device)
            # z = Variable(torch.randn(imgs.shape[0], opt.latent_dim, 1, 1)).to(device=device)
        policy.set_parameters(parent_param, generator)
        if random.random() > 0.1:
            
            generator.zero_grad()

            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), label_G.fill_(pop_size))
            label_D = Variable(torch.LongTensor(imgs.shape[0]//pop_size),requires_grad=False).to(device)
            fake_loss = adversarial_loss(discriminator(gen_imgs[:imgs.shape[0]//pop_size].detach()), label_D.fill_(j))
            (g_loss+fake_loss).backward()
            torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)).step()
            rollout_reward = g_loss.item()
            #chid_param = self.policy.get_parameters(generator)
        else:
            
            with torch.no_grad():
                for param in generator.parameters():
                    param.add_(torch.randn(param.size()) * 0.001)
            gen_imgs = generator(z)
            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), label_G.fill_(pop_size))
            rollout_reward = g_loss.item()

        return {"child": policy.get_parameters(generator), "rollout_reward": rollout_reward, "fake":gen_imgs, "index":j,}

# ray.init()

policy = Policy(pop_size)
workers = [Worker()
            for _ in range(num_workers)]

writer_Loss_EGAN_softmax = SummaryWriter(f'runs/Loss_EGAN_softmax_p{pop_size}_f{fifth}')
writer_real = SummaryWriter(f'runs/EGAN_softmax_MNIST_p{pop_size}_f{fifth}/test_real')
writer_fake = SummaryWriter(f'runs/EGAN_softmax_MNIST_p{pop_size}_f{fifth}/test_fake')


# ----------
#  Training
# ----------


for epoch in range(opt.n_epochs):
    for i, (imgs, ) in enumerate(dataloader):
        
        #valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor)).to(device)
        #z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
        t1 = time.time()
        # -----------------
        #  Train Generator
        # -----------------
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)

        rollout_ids = []
        results = []
        

        for j in range(pop_size):
            parent_generator = policy.get_parameters(generator[j])
            # generator_id = ray.put(generator[j])
            # #z_id = ray.put(z)
            # discriminator_id = ray.put(discriminator)

            # Use the actors to do rollouts,
            # note that we pass in the ID of the policy weights.
            
            results += [workers[j*fifth + f].do_rollouts(z,
                          parent_generator,  generator[j], discriminator,  j) for f in range (fifth)]
            # Get the results of the rollouts.
        # results = ray.get(rollout_ids)
        # Loop over the results.
        #results =  sorted(results, key=lambda x:x['rollout_reward'])
        all_rollout_rewards, population, pictures, index_list = [], [], [], []
        for result in results:
            all_rollout_rewards.append(result["rollout_reward"])
            population.append(result["child"])
            pictures.append(result["fake"])
            index_list.append(result["index"])
        children, gen_imgs, g_loss, best_picture =  policy.update(all_rollout_rewards, population, pictures,index_list)
        for i_g in range(len(generator)):
            policy.set_parameters(children[i_g], generator[i_g])
        #print(time.time()- t1)
        
        # ---------------------
        #  Train Discriminator
        # ---------------------
        # Measure discriminator's ability to classify real from generated samples
        optimizer_D.zero_grad()
        label_D = Variable(torch.LongTensor(imgs.shape[0]),requires_grad=False).to(device)
        real_loss = adversarial_loss(discriminator(real_imgs), label_D.fill_(pop_size))
        for x in range(pop_size):
            
            label_D = Variable(torch.LongTensor(imgs.shape[0]//pop_size),requires_grad=False).to(device)
            fake_loss = adversarial_loss(discriminator(gen_imgs[x][:imgs.shape[0]//pop_size].detach()), label_D.fill_(x))
            real_loss += (fake_loss)
        (real_loss/2).backward()
        optimizer_D.step()

        writer_Loss_EGAN_softmax.add_scalars(
            'Loss_GAN',
            {'Loss_G':g_loss,'Loss_D':real_loss.item()}, 
            epoch * len(dataloader) + i)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(dataloader), real_loss.item(), g_loss)
        )

        batches_done = epoch * len(dataloader) + i
        # if batches_done % opt.sample_interval == 0:
        #     img_grid_real = torchvision.utils.make_grid(real_imgs.data[:25], nrow=5,normalize=True)
        #     img_grid_fake = torchvision.utils.make_grid(best_picture.data[:25], nrow=5,normalize=True)
        #     writer_real.add_image('Real images', img_grid_real)
        #     writer_fake.add_image('Fake images', img_grid_fake)
        #     save_image(best_picture.data[:25], "fake_images/%d.png" % batches_done, nrow=5, normalize=True)
        #     save_image(real_imgs.data[:25], "real_images/%d.png" % batches_done, nrow=5, normalize=True)
        #     policy.save_model(generator, discriminator)
    # fid, is_score = testFID.test_fid_is(generator[0])
    # with open(f'{pop_size}_{fifth}_fid.txt', 'a') as f:
    #     f.write(f'{epoch},{fid},{is_score[0]}\n')

    # for g in generator:
    #     g.eval()
    if epoch % 100 == 0:
        plt.clf()
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
        plt.scatter(X[:, 0], X[:, 1])

        for i in range(len(generator)):
            fake = generator[i](z).data

            plt.scatter(fake[:, 0], fake[:, 1])
        # plt.ion()
        plt.show(block=False)
        plt.pause(0.001)
plt.clf()
z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim)))).to(device)
plt.scatter(X[:, 0], X[:, 1])

for i in range(len(generator)):
    fake = generator[i](z).data

    plt.scatter(fake[:, 0], fake[:, 1])

plt.show()            
