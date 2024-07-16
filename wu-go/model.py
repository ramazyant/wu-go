import numpy as np
import pandas as pd

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

# cuda = True if torch.cuda.is_available() else False
Tensor = torch.FloatTensor #torch.cuda.FloatTensor if cuda else 


class Generator(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_size=64):
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_outputs),
        )

    def forward(self, x_cond, x_noise):
        x = torch.cat((x_cond, x_noise), dim=1)
        x = self.model(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, n_inputs, hidden_size=64):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.model(x)
        return x


# Gradient pealty
def compute_gp(D, cond_data, real_test_data, gen_data):

    # Random weight term for interpolation between real and fake samples
    alpha = Tensor(np.random.random((real_test_data.size(0), 1)))

    # Get random interpolation between real and fake samples
    inter = (alpha * real_test_data + ((1 - alpha) * gen_data)).requires_grad_(True)
    D_inter = D(torch.cat((cond_data, inter), dim=1))
    fake = Variable(Tensor(real_test_data.size(0), 1).fill_(1.0))

    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs      = D_inter,
        inputs       = inter,
        grad_outputs = fake,
        create_graph = True,
        retain_graph = True,
        only_inputs  = True,
    )[0]
    
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gp

    
class WGAN_GP(object):
    def __init__(self, gan_type='W', batch_size=16, hidden_size=64, n_epochs=100, n_disc=5, G_lr=1e-3, D_lr=1e-3, latent_dim=10, scaling=False, lambda_gp=1, scheduler=False):
        
        self.gan_type    = gan_type
        self.batch_size  = batch_size
        self.hidden_size = hidden_size
        self.n_epochs    = n_epochs
        self.n_disc      = n_disc
        self.latent_dim  = latent_dim
        self.scaling     = scaling
        self.lambda_gp   = lambda_gp
        self.G_lr        = G_lr
        self.D_lr        = D_lr
        self.scheduler   = scheduler
        self.G_loss_hist = []
        self.D_loss_hist = []
        
        
    def fit(self, X, X_cond):
        
        X = Tensor(X)
        X_cond = Tensor(X_cond)
        X_train = TensorDataset(X, X_cond)
        
        if self.scheduler:
            self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.G_lr*1e2)
            self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.D_lr*1e2)
            
            G_scheduler = StepLR(self.G_opt, step_size=self.n_epochs//3, gamma=0.1)
            D_scheduler = StepLR(self.D_opt, step_size=self.n_epochs//3, gamma=0.1)
        else:
            self.G_opt = torch.optim.Adam(self.G.parameters(), lr=self.G_lr)
            self.D_opt = torch.optim.Adam(self.D.parameters(), lr=self.D_lr)
        
        self.G.train(True)
        self.D.train(True)
        
        progress_bar = tqdm(range(self.n_epochs), unit="epoch")
        for e in progress_bar:
            for X_batch, X_cond_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True):

                for _ in range(self.n_disc):
                    noise_batch = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))

                    gen_batch = self.G(X_cond_batch, noise_batch)

                    D_real = self.D(torch.cat((X_cond_batch, X_batch), dim=1))
                    D_fake = self.D(torch.cat((X_cond_batch, gen_batch.detach()), dim=1))
                    
                    D_loss = torch.mean(D_fake) - torch.mean(D_real)
                    grad_p = compute_gp(self.D, X_cond_batch, X_batch, gen_batch.detach())
                    D_loss += self.lambda_gp * grad_p

                    self.D_opt.zero_grad()
                    D_loss.backward()
                    self.D_opt.step()
                    
                    self.D_loss_hist.append(D_loss.detach().numpy())
                
                noise_batch = Variable(Tensor(np.random.normal(0, 1, (self.batch_size, self.latent_dim))))

                gen_batch = self.G(X_cond_batch, noise_batch)
                
                D_fake = self.D(torch.cat((X_cond_batch, gen_batch), dim=1))
                
                G_loss = -torch.mean(D_fake)
            
                self.G_opt.zero_grad()
                G_loss.backward()
                self.G_opt.step()
                
                self.G_loss_hist.append(G_loss.detach().numpy())
            
            progress_bar.set_description(f"G loss: {G_loss:.4f}, D loss: {D_loss:.4f}")
            
            if self.scheduler:
                G_scheduler.step()
                D_scheduler.step()
            
        self.G.train(False)
        self.D.train(False)
    
    
    def generate(self, X_cond):
        noise = Variable(Tensor(np.random.normal(0, 1, (len(X_cond), self.latent_dim))))
        X_gen = self.G(Tensor(X_cond).detach(), noise.detach())
        return X_gen.detach().numpy()
    
    
    def _scale(self, X):
        
        ss = StandardScaler()
        ss.fit(X)
        X_ss = ss.transform(X)
        
        return X_ss
    
    
    def predict(self, X, scheduler=False):
        
        if self.scaling:
            X = self._scale(X)
        
        X, X_cond = X[:, :1], X[:, 1:]

        if X_cond.shape[-1] == 1:
            X_cond = X_cond.reshape(-1, 1)

        self.scheduler = scheduler
        
        self.fit(X, X_cond)
        
        X_gen = self.generate(X_cond)
        
        return X_gen
