import gc
import time
import torch
import pickle
import numpy as np
from ego_models import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import qmc, norm
from matplotlib import cm, ticker
from IPython.display import Image
import matplotlib.animation as animation
from scipy.stats import wasserstein_distance as Wdist


class EGO(object):
    def __init__(self, model, func, sampling='olhs', n_samples=4, max_iter=100, sigma=1e-1, n_obs=int(1e3), n_inter=int(1e3), n_splits=100, opt_eps=1e-1, seed=None, plot_res=False, opt='EI', kappa=2.0):
        
        self.model     = model
        self.func      = func
        self.sampling  = sampling
        self.lhs_ort   = 1 if sampling == 'lhs' else 2
        self.max_iter  = max_iter
        self.n_obs     = n_obs
        self.n_inter   = n_inter
        self.sigma     = sigma
        self.olhs_seed = seed
        self.plot_res  = plot_res
        self.opt       = opt
        self.n_splits  = n_splits
        self.kappa     = kappa
        
        if (self.func.dim > 2) and (sampling == 'lhs'):
            self.n_samples = 25
        else:
            self.n_samples = n_samples
            
        if self.func.name == 'rosenbrock20':
            self.opt_eps = 2.0
        elif self.func.name == 'rosenbrock8':
            self.opt_eps = 1.1
            self.n_samples = 121
        elif self.func.name == 'tang':
            self.opt_eps = 5.0
        else:
            self.opt_eps = opt_eps
    

    def get_conds(self, N=None):

        if N is None:
            N = self.n_splits
        elif self.func.dim > 2:
            N = 4
        
        X = np.meshgrid(*[np.linspace(self.func.domain[i, 0], self.func.domain[i, 1], N) for i in range(self.func.dim)])
        for i in range(self.func.dim):
            X[i] = X[i].flatten()
        return np.array(X).T
    
        
    def toy_example(self, conds=None, rand_cond=False, n_subsamples=None):
        
        if n_subsamples is None: n_subsamples = self.n_samples
        
        if (conds is None) and ('lhs' in self.sampling):
            
            lhs   = qmc.LatinHypercube(d=self.func.dim, scramble=True, strength=self.lhs_ort, seed=self.olhs_seed)
            conds = lhs.random(self.n_samples)
            conds = qmc.scale(conds, self.func.domain[:, 0], self.func.domain[:, 1])
        
        f_conds = self.func(conds)
        
        if n_subsamples == 1:
            f_conds = [f_conds]
        
        for i in range(len(conds)):
            
            if self.func.name == 'levi':
                s1 = 0.04 - 0.03 * np.square(np.sin(3 * np.pi * conds[i, 1]))
                s2 = 0.001 + 0.03 * np.square(np.sin(3 * np.pi * conds[i, 1]))
                g1 = np.random.normal(f_conds[i]-0.05, s1, (self.n_obs//2, 1))
                g2 = np.random.normal(f_conds[i]+0.05, s2, (self.n_obs//2, 1))
                gauss = np.concatenate((g1, g2), axis=0)
                gauss = np.concatenate((gauss, conds[i]*np.ones((self.n_obs, 1))), axis=1)
            else:
                gauss = np.concatenate((np.random.normal(f_conds[i], self.sigma, (self.n_obs, 1)), conds[i]*np.ones((self.n_obs, 1))), axis=1)
        
            if i: sample = np.concatenate((sample, gauss))
            else: sample = gauss
        
        return sample, conds
    
    
    def get_inter_conds(self, conds):
        values = np.linspace(min(conds), max(conds), (len(conds) - 1) * self.n_splits + 1)
        points = np.arange(len(values))
        xi = points + 0.5
        res = scipy.interpolate.griddata(points=points, values=values, xi=xi[:-1], method='linear', fill_value=-1)
        return np.setdiff1d(res, [min(conds), max(conds)])
    
    
    def get_inter(self):
        
        if self.func.dim == 2:
            conds = self.get_conds()
        else:
            lhs   = qmc.LatinHypercube(d=self.func.dim, scramble=True, strength=2, seed=self.olhs_seed)
            conds = lhs.random(10201)
            conds = qmc.scale(conds, self.func.domain[:, 0], self.func.domain[:, 1])
        
        f_conds = self.func(conds)

        inter_true, inter_conds = [], []

        for i in range(len(conds)):
            inter_true.append(np.random.normal(f_conds[i], self.sigma, (self.n_inter, 1)))
            inter_conds.append(conds[i] * np.ones((self.n_inter, 1)))
        
        inter_true, inter_conds = np.array(inter_true), np.array(inter_conds)
        
        return np.concatenate((inter_true, inter_conds), axis=-1), conds
    
    
    def optimise(self):
        
        res = {'inter_conds':[], 'conds':[], 'std':[], 'EI':[], 'f':[], 'f_true':[]}
        
        for n_iter in range(self.max_iter+1):
            
            print(f'iteration {n_iter + 1}')
            
            if n_iter == 0: X, conds = self.toy_example()
            else:
                X__, conds__ = self.toy_example(conds=best_next_params, n_subsamples=1)
                
                X = np.concatenate((X, X__), axis=0)
                conds = np.unique(X[:, 1:], axis=0)
            
            print('Fitting the model...')
            
            if (n_iter == 0) and ('GP' in str(self.model)):
                self.model = EGO_model(self.model, X=X[:, 1:], Y=X[:, :1])
            
            self.model.fit(X)
            
            if n_iter == 0:
                inter_true, inter_conds = self.get_inter()
                f_true = np.mean(inter_true, axis=1)[:, 0].reshape(-1, 1)
            
            if self.model.name == 'GAN':
                X_train = X.reshape(self.n_samples+n_iter, self.n_obs, X.shape[-1])
                f, std = self.model.predict(inter_conds, X_train=X_train)
                EI = f - self.kappa * std
                
            elif self.opt == 'EI':
                f, std = self.model.predict(inter_conds)
                I  = np.maximum(np.zeros(f.shape), np.min(self.func(conds)) - f)
                Z  = I / (std + 1e-8)
                EI = I * norm.cdf(Z) + std * norm.pdf(Z)
                
            else:
                f, std = self.model.predict(inter_conds)
                EI = f - self.kappa * std
            
            EI = EI.reshape(-1, 1)
            
            if self.plot_res:
                self.plot(conds=conds, inter_conds=inter_conds, f_true=f_true, f=f, std=std, EI=EI)
            
            res['std'].append(std)
            res['EI'].append(EI)
            res['f'].append(f)
            
            if n_iter == 0: # to save some memory
                res['inter_conds'].append(inter_conds)
                res['conds'].append(conds)
                res['f_true'].append(f_true)
            else:
                res['inter_conds'].append(None)
                res['conds'].append(conds__)
                res['f_true'].append(None)
            
            self.model.n_iter = n_iter + 1
            
            if self.opt == 'EI': best_next_params = np.array([inter_conds[np.argmax(EI)]])
            else: best_next_params = np.array([inter_conds[np.argmin(EI)]])
            
            if np.any(np.linalg.norm(self.func.glob_min - inter_conds[np.argmin(EI)], ord=2, axis=-1) < self.opt_eps):
                print(f'{self.model.name} has converged!')
                break
            elif n_iter == self.max_iter:
                print(f'{self.model.name} has failed...')
                break
            
            del f
            del EI
            del std
            gc.collect()
            
        return res
    
    
    def plot(self, **kwargs):
        
        fig, ax = plt.subplots(ncols=4, figsize=(20, 5))
        
        conds       = kwargs['conds']
        inter_conds = kwargs['inter_conds']
        
        f_true = kwargs['f_true'].reshape(self.n_splits, self.n_splits)
        f      = kwargs['f'].reshape(self.n_splits, self.n_splits)
        std    = kwargs['std'].reshape(self.n_splits, self.n_splits)
        EI     = kwargs['EI'].reshape(self.n_splits, self.n_splits)

        ax[0].contourf(f_true, cmap=cm.RdBu, locator=ticker.LinearLocator(), extent=self.func.domain.flatten())
        for glob_min in self.func.glob_min:
            ax[0].scatter(glob_min[0], glob_min[1], color='red', marker='*')
        ax[0].scatter(conds[:, 0], conds[:, 1], color='black')
        ax[0].set_title('True function')

        ax[1].contourf(f, cmap=cm.RdBu, locator=ticker.LinearLocator(), extent=self.func.domain.flatten())
        for glob_min in self.func.glob_min:
            ax[1].scatter(glob_min[0], glob_min[1], color='red', marker='*')
        ax[1].scatter(conds[:, 0], conds[:, 1], color='black')
        ax[1].set_title('Estimated function')

        ax[2].contourf(std, cmap=cm.RdBu, locator=ticker.LinearLocator(), extent=self.func.domain.flatten())
        for glob_min in self.func.glob_min:
            ax[2].scatter(glob_min[0], glob_min[1], color='red', marker='*')
        ax[2].scatter(conds[:, 0], conds[:, 1], color='black')
        ax[2].set_title(r'$\sigma(x)$')

        max_ei = inter_conds[np.argmax(EI)]
        ax[3].contourf(EI, cmap=cm.RdBu, locator=ticker.LinearLocator(), extent=self.func.domain.flatten())
        for glob_min in self.func.glob_min:
            ax[3].scatter(glob_min[0], glob_min[1], color='red', marker='*')
        ax[3].scatter(conds[:, 0], conds[:, 1], color='black')
        ax[3].scatter(max_ei[0], max_ei[1], color='turquoise', marker='*')
        ax[3].set_title('Expected Improvement')

        plt.show()
