import gc
import GPy
import torch
import deepgp
import pickle
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torchbnn as bnn
import torch.nn.functional as F
from matplotlib import cm, ticker
from IPython.display import Image
from model import WGAN_GP, Generator, Discriminator
from scipy.stats import energy_distance as Wdist
from torchensemble import AdversarialTrainingRegressor
from torch.utils.data import TensorDataset, DataLoader
from GPy.core.parameterization.variational import NormalPosterior

Tensor = torch.FloatTensor


class Regressor(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, hidden_size=16):
        super(Regressor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(n_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, n_outputs),
        )

    def forward(self, x):
        x = self.model(x)
        return x


class BayesianRegressor(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, hidden_size=64):
        super(BayesianRegressor, self).__init__()
        
        self.model = nn.Sequential(
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=n_inputs, out_features=hidden_size),
            nn.Tanh(),
            bnn.BayesLinear(prior_mu=0, prior_sigma=0.1, in_features=hidden_size, out_features=n_outputs),
        )
    
    def forward(self, x):
        x = self.model(x)
        return x


class EGO_model(object):
    def __init__(self, model='GP', **kwargs):
        
        self.n_iter     = 0
        self.n_inter    = kwargs['n_inter'] if 'n_inter' in kwargs else int(1e3)
        self.name       = model
        self.lr         = kwargs['lr'] if 'lr' in kwargs else 1e-3
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 16
        

        if model != 'DE':
            self.n_epochs = kwargs['n_epochs'] if 'n_epochs' in kwargs else 100
        else:
            self.n_epochs = kwargs['n_epochs'] if 'n_epochs' in kwargs else 20
        
        
        if model == 'GP':
            
            X, Y       = kwargs['X'], kwargs['Y']
            self.model = GPy.models.GPRegression(X, Y)
            
        elif model == 'DE':
            
            base_estimator = Regressor(n_inputs=kwargs['n_inputs'] if 'n_inputs' in kwargs else 2)
            
            self.model = AdversarialTrainingRegressor(
                            estimator    = base_estimator,
                            n_estimators = kwargs['n_estimators'] if 'n_estimators' in kwargs else 10)
            
            self.model.device = kwargs['device'] if 'device' in kwargs else torch.device('cpu')
            
            self.model.set_optimizer(kwargs['opt'] if 'opt' in kwargs else 'Adam', lr = self.lr,
                                     weight_decay = kwargs['weight_decay'] if 'weight_decay' in kwargs else 0)
            
            
            if 'scheduler' in kwargs:
                self.model.set_scheduler('StepLR',
                                        step_size = kwargs['sched_step'] if 'sched_step' in kwargs else self.n_epochs//3,
                                        gamma = kwargs['sched_gamma'] if 'sched_gamma' in kwargs else 0.1)
            
        elif model == 'DGP':
            
            self.hidden = kwargs['hidden'] if 'hidden' in kwargs else 10
            
        elif model == 'BNN':
            
            self.model     = BayesianRegressor(n_inputs=kwargs['n_inputs'] if 'n_inputs' in kwargs else 2)
            self.mse_loss  = nn.MSELoss()
            self.kl_loss   = bnn.BKLLoss(reduction='mean', last_layer_only=False)
            self.kl_weight = kwargs['kl_weight'] if 'kl_weight' in kwargs else 1e-2
            self.opt       = torch.optim.Adam(self.model.parameters(), lr=self.lr)
            self.n_preds   = kwargs['n_preds'] if 'n_preds' in kwargs else 10
        
        elif model == 'GAN':
            
            self.verbose = kwargs['verbose'] if 'verbose' in kwargs else False
            
            self.model = WGAN_GP(n_epochs = self.n_epochs,
                                 n_disc   = kwargs['n_disc'] if 'n_disc' in kwargs else 5,
                                 G_lr     = kwargs['G_lr'] if 'G_lr' in kwargs else 1e-4,
                                 D_lr     = kwargs['D_lr'] if 'D_lr' in kwargs else 1e-4)
            
            n_inputs = kwargs['n_inputs'] if 'n_inputs' in kwargs else 2
            
            self.model.G = Generator(n_inputs   = self.model.latent_dim + n_inputs,
                                     n_outputs  = 1, hidden_size = self.model.hidden_size)
            
            self.model.D = Discriminator(n_inputs = n_inputs + 1, hidden_size = self.model.hidden_size)
    
        
    def fit(self, X):
        
        if self.name == 'GP':
            
            if self.n_iter > 0:
                self.model.set_XY(X=X[:, 1:], Y=X[:, :1])
                
            self.model.optimize(messages=True, max_iters=self.n_epochs)
            
        elif self.name == 'DE':
            
            c, x = Tensor(X[:, 1:]), Tensor(X[:, :1])
            c -= torch.min(c)
            c /= torch.max(c)
            
            X_train = TensorDataset(c, x)
            
            train_loader = DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.model.fit(train_loader, epochs=self.n_epochs, log_interval=1000, save_model=False)
            
        elif self.name == 'DGP':
            
            X, Y = X[:, 1:], X[:, :1]

#             if self.n_iter != 0:
#                 del self.model
#                 gc.collect()
            
            self.model = deepgp.DeepGP([Y.shape[1], self.hidden, X.shape[1]], Y=Y, X=X,
                                       kernels=[GPy.kern.RBF(self.hidden, variance=1e-1), GPy.kern.RBF(X.shape[1], variance=1e-1)],
                                       num_inducing=self.n_epochs, back_constraint=False)
            
            self.model.optimize(messages=True, max_iters=self.n_epochs)
            
        elif self.name == 'BNN':
            
            X_train = TensorDataset(Tensor(X[:, :1]), Tensor(X[:, 1:]))
            
            for _ in tqdm(range(self.n_epochs), unit='epochs'):
                for X_batch, cond_batch in DataLoader(X_train, batch_size=self.batch_size, shuffle=True, drop_last=True):
                    
                    y_pred   = self.model(cond_batch)
                    mse_loss = self.mse_loss(y_pred, X_batch)
                    kl_loss  = self.kl_loss(self.model)
                    loss     = mse_loss + self.kl_weight * kl_loss

                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
        
        elif self.name == 'GAN':
            
            if self.n_iter == 0: X_gen = self.model.predict(X, scheduler=False)
            else:                X_gen = self.model.predict(X, scheduler=True)
    
    
    def predict(self, inter_conds, X_train=None):
        
        f, std = -1, -1
        
        if self.name == 'GP':
            
            f, std = self.model.predict(inter_conds)
            std = np.sqrt(std)
            
        elif self.name == 'DE':
            
            inter_conds = Tensor(inter_conds)
            inter_conds -= torch.min(inter_conds)
            inter_conds /= torch.max(inter_conds)
            
            f = []
            
            for estimator in self.model.estimators_:
                f.append(estimator(inter_conds).detach().numpy())
            
            f   = np.concatenate(f, axis=-1)
            std = np.std(f, axis=-1)
            f   = np.mean(f, axis=-1)
            
        elif self.name == 'DGP':
            
            f, std = self.model.predict(inter_conds)
            std = np.sqrt(std)
            
        elif self.name == 'BNN':
            
            f = []
            
            for _ in range(self.n_preds):
                f.append(self.model(Tensor(inter_conds)).detach().numpy())

            f   = np.concatenate(f, axis=-1)
            std = np.std(f, axis=-1)
            f   = np.mean(f, axis=-1)
        
        elif self.name == 'GAN':
            
            print('Calculating WU...')
            
            inter_gen = np.array([self.model.generate(c * np.ones((self.n_inter, 1))) for c in inter_conds])

            W, rng = [], range(inter_gen.shape[0])
            
            if self.verbose: rng = tqdm(rng)
                
            for i in rng:
                W.append([Wdist(X_train[j, :, 0], inter_gen[i, :, 0]) for j in range(X_train.shape[0])])#+self.n_iter
            W = np.min(np.array(W), axis=-1)

            f, std = np.mean(inter_gen, axis=1)[:, 0], W
            
        return f, std

