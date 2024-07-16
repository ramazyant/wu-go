import warnings
warnings.filterwarnings("ignore")

import os
import time
import scipy
import torch
import numpy as np
from model import *
import seaborn as sns
from scipy.stats import kde
import matplotlib.pyplot as plt
from IPython.display import Image
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
from sklearn.metrics import roc_auc_score
from scipy.spatial import cKDTree as KDTree
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

global random_seeds
random_seeds = [2, 3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 22, 42, 142, 737] # 30 random seeds


def func_transform(conds, func='linear', alpha=10, k=2, mu=0):
    
    if func == 'linear':
        f_conds = alpha * (mu + conds)
    elif func == 'parabolic':
        f_conds = (mu + conds) ** k
    elif func == 'hyperbolic':
        f_conds = k ** (mu + conds)
    elif func == 'xsinx':
        f_conds = (mu + conds) * np.sin((mu + conds))
    else:
        return -1
    
    return f_conds


def toy_example_2d(n_subsamples=3, n_obs=int(1e3), c_min=0, c_max=10, conds = None, mu=0, sigma=1, func='linear', alpha=10, k=2, rand_cond=False):
    
    if conds is None:
        conds = np.arange(c_min, c_max+1e-5, (c_max - c_min)/(n_subsamples - 1))
    
    f_conds = func_transform(conds, func=func, alpha=alpha, k=k, mu=mu)
    
    for i in range(len(conds)):
        
        gauss = np.concatenate((np.random.normal(f_conds[i], sigma, (n_obs, 1)), conds[i]*np.ones((n_obs, 1))), axis=1)
            
        if i: toy = np.concatenate((toy, gauss))
        else: toy = gauss
    
    idx = np.random.permutation(len(toy))
    if rand_cond:
        toy += np.c_[np.zeros(len(toy)), np.random.normal(0, 0.1, len(toy))]
    
    return np.array(toy[idx]), conds.tolist()


def get_inter_conds(c, n_splits=5):
    values = np.linspace(min(c), max(c), (len(c)-1)*n_splits + 1)
    points = np.arange(len(values))
    xi = points + 0.5
    res = scipy.interpolate.griddata(points=points, values=values, xi=xi[:-1], method='linear', fill_value=-1)
    return np.setdiff1d(res, [min(c), max(c)])


def get_inter(conds, model, func=None, n_obs=int(1e3), mu=0, sigma=1, alpha=10, k=2, n_splits=5):
    
    conds   = get_inter_conds(conds, n_splits=n_splits)
    f_conds = func_transform(conds, func=func, alpha=alpha, k=k, mu=mu)
        
    inter_true, inter_conds = [], []
        
    if func != None:
        for i in range(len(conds)):
            inter_true.append(np.random.normal(f_conds[i], sigma, (n_obs, 1)))
            inter_conds.append(conds[i] * np.ones((n_obs, 1)))
    else:
        for c in conds:
            inter_true.append((np.random.normal(mu + c, sigma, (n_obs, 1))))
    
    inter_gen = [model.generate(c * np.ones((n_obs, 1))) for c in conds]
    
    inter_true, inter_gen, inter_conds  = np.concatenate(inter_true), np.concatenate(inter_gen), np.concatenate(inter_conds)
    inter_gen = inter_gen.reshape(-1, 1)
    
    return np.concatenate((inter_true, inter_conds), axis=1), np.concatenate((inter_gen, inter_conds), axis=1), conds.tolist()


def kl_divergence(x, y):
    """
        Compute the Kullback-Leibler divergence between two multivariate samples.
        Parameters
        ----------
        x : 2D array (n,d)
        Samples from distribution P, which typically represents the true
        distribution.
        y : 2D array (m,d)
        Samples from distribution Q, which typically represents the approximate
        distribution.
        Returns
        -------
        out : float
        The estimated Kullback-Leibler divergence D(P||Q).
        References
        ----------
        Pérez-Cruz, F. Kullback-Leibler divergence estimation of
        continuous distributions IEEE International Symposium on Information
        Theory, 2008.
    """
    
    # Check the dimensions are consistent
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)

    n,d = x.shape
    m,dy = y.shape

    assert(d == dy)

    # Build a KD tree representation of the samples and find the nearest neighbour
    # of each point in x.
    xtree = KDTree(x)
    ytree = KDTree(y)

    # Get the first two nearest neighbours for x, since the closest one is the
    # sample itself.
    r = xtree.query(x, k=2, eps=.01, p=2)[0][:,1]
    s = ytree.query(x, k=1, eps=.01, p=2)[0]

    # There is a mistake in the paper. In Eq. 14, the right side misses a negative sign
    # on the first term of the right hand side.
    return -np.log(r/s).sum() * d / n + np.log(m / (n - 1.))


def classify(X, n_inter):
    
    y = np.concatenate((np.zeros(n_inter), np.ones(n_inter)))
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    
    lda = LDA()
    lda.fit(X_train, y_train)
    y_score = lda.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, y_score)
    
    
def qm_ci_test(n_tests=30, n_obs=int(1e4), n_inter=int(1e4), n_splits=100, n_samples=3, func='parabolic', c_min=1, c_max=13,
                n_epochs=100, batch_size=16, lambda_gp=1, n_disc=5, G_lr=1e-4, D_lr=1e-4):
    
    kls, rocs = [], []
    
    for n in range(n_tests):
        
        print(f'Interpolation test #{n}')
        
        np.random.seed(random_seeds[n])
        
        # Creating toy example
        X, conds = toy_example_2d(n_subsamples=n_samples, n_obs=n_obs, c_min=c_min, c_max=c_max, func=func)

        # Fitting the model
        model = WGAN_GP(n_epochs=n_epochs, batch_size=batch_size, lambda_gp=lambda_gp, n_disc=n_disc, G_lr=G_lr, D_lr=D_lr)

        # Generating sample
        X_gen = model.predict(X)

        # Getting interpolation true and generated samples samples
        inter_true, inter_gen, inter_conds = get_inter(conds, model, func=func, n_obs=n_inter, n_splits=n_splits)
        
        one_test_kls, one_test_rocs = [], []
        
        for i in range(inter_true.shape[0] // n_inter):
            
            X_true = inter_true[i*n_inter : (i+1)*n_inter, :]
            X_gen = inter_gen[i*n_inter : (i+1)*n_inter, :]
            
            one_test_kls.append(kl_divergence(X_true, X_gen))
            one_test_rocs.append(classify(np.concatenate((X_true, X_gen)), n_inter))
        
        kls.append(one_test_kls)
        rocs.append(one_test_rocs)
    
    fig, ax = plt.subplots(ncols=2, figsize=(12, 6))
    
    mean_kl = np.mean(kls, axis=0)
    ci_kl   = 1.96 * np.std(kls, axis=0)/np.sqrt(len(mean_kl))
    
    ax[0].plot(inter_conds, mean_kl)
    ax[0].fill_between(inter_conds, mean_kl - ci_kl, mean_kl + ci_kl, color='blue', alpha=0.1)
    ax[0].vlines(conds, ymin=0, ymax=max(mean_kl)+0.1, colors='black')
    ax[0].set_title('Pairwise KL divergence')
    
    mean_roc = np.mean(rocs, axis=0)
    ci_roc   = 1.96 * np.std(rocs, axis=0)/np.sqrt(len(mean_roc))
    
    ax[1].plot(inter_conds, mean_roc)
    ax[1].fill_between(inter_conds, mean_roc - ci_roc, mean_roc + ci_roc, color='blue', alpha=0.1)
    ax[1].vlines(conds, ymin=min(0.5, min(mean_roc)), ymax=1+1e-5, colors='black')
    ax[1].hlines(0.5, xmin=min(inter_conds), xmax=max(inter_conds), colors='red')
    ax[1].set_title('Pairwise classification ROC AUC')
    
    current_time = time.strftime("%d.%m-%H:%M:%S")
    plt.savefig(f'plots/qm_ci_test_{current_time}_{func}_{n_samples}_{c_min}_{c_max}.jpeg', format='jpeg')
    
    plt.show()


class sfg():
    def __init__(self, seaborngrid, fig, subplot_spec, title='abcd'):
        self.fig = fig
        self.sg = seaborngrid
        self.subplot = subplot_spec
        self.title = title
        if isinstance(self.sg, sns.axisgrid.FacetGrid) or \
            isinstance(self.sg, sns.axisgrid.PairGrid):
            self._movegrid()
        elif isinstance(self.sg, sns.axisgrid.JointGrid):
            self._movejointgrid()
        self._finalize()

    def _movegrid(self):
        """ Move PairGrid or Facetgrid """
        self._resize()
        n = self.sg.axes.shape[0]
        m = self.sg.axes.shape[1]
        self.subgrid = gridspec.GridSpecFromSubplotSpec(n,m, subplot_spec=self.subplot)
        for i in range(n):
            for j in range(m):
                
                self._moveaxes(self.sg.axes[i,j], self.subgrid[i,j])

    def _movejointgrid(self):
        """ Move Jointgrid """
        h= self.sg.ax_joint.get_position().height
        h2= self.sg.ax_marg_x.get_position().height
        r = int(np.round(h/h2))
        self._resize()
        self.subgrid = gridspec.GridSpecFromSubplotSpec(r+1,r+1, subplot_spec=self.subplot)
        
        self.sg.ax_marg_x.set_title(self.title)
        
        self._moveaxes(self.sg.ax_joint, self.subgrid[1:, :-1])
        self._moveaxes(self.sg.ax_marg_x, self.subgrid[0, :-1])
        self._moveaxes(self.sg.ax_marg_y, self.subgrid[1:, -1])

    def _moveaxes(self, ax, gs):
        #https://stackoverflow.com/a/46906599/4124317
        ax.remove()
        ax.figure=self.fig
        self.fig.axes.append(ax)
        self.fig.add_axes(ax)
        ax._subplotspec = gs
        ax.set_position(gs.get_position(self.fig))
        ax.set_subplotspec(gs)

    def _finalize(self):
        plt.close(self.sg.fig)
        self.fig.canvas.mpl_connect("resize_event", self._resize)
        self.fig.canvas.draw()

    def _resize(self, evt=None):
        self.sg.fig.set_size_inches(self.fig.get_size_inches())


def inter_test(n_obs=int(1e3), n_inter=int(1e3), n_splits=100, n_samples=3, func='linear', c_min=1, c_max=13, k=2, sigma=1, rand_cond=False,
               get_qm=False, profiles=False, n_epochs=100, batch_size=16, lambda_gp=1, n_disc=5, G_lr=1e-4, D_lr=1e-4):
    
    np.random.seed(73)
    
    # Creating toy example
    X, conds = toy_example_2d(n_subsamples=n_samples, n_obs=n_obs, c_min=c_min, c_max=c_max, func=func, k=k, sigma=sigma, rand_cond=rand_cond)
    
    # Creating the model
    if not os.listdir(f'baseline_gan/{func}'):
        print('Fitting Baseline GAN...')
        model = WGAN_GP(n_epochs=n_epochs, batch_size=batch_size, lambda_gp=lambda_gp, n_disc=n_disc, G_lr=G_lr, D_lr=D_lr)
        # Fitting the model and Generating sample
        X_gen = model.predict(X)
        # Save model
        torch.save(model.G.state_dict(), f'baseline_gan/{func}/G.pth')
        torch.save(model.D.state_dict(), f'baseline_gan/{func}/D.pth')
        
    else:
        X_real, X_cond = X[:, :-1], X[:, -1].reshape(-1, 1)
        model = WGAN_GP(n_epochs=n_epochs, batch_size=batch_size, lambda_gp=lambda_gp, n_disc=n_disc, G_lr=G_lr, D_lr=D_lr)
        model.G = Generator(n_inputs=model.latent_dim + X_cond.shape[1], n_outputs=X_real.shape[1], hidden_size=model.hidden_size)
        model.D = Discriminator(n_inputs=X_real.shape[1] + X_cond.shape[1], hidden_size=model.hidden_size)
            
        model.G.load_state_dict(torch.load(f'baseline_gan/{func}/G.pth'))
        model.D.load_state_dict(torch.load(f'baseline_gan/{func}/D.pth'))
        model.G.eval()
        model.D.eval()
        print('Baseline loaded')
    
    # Getting interpolation true and generated samples samples
    inter_true, inter_gen, inter_conds = get_inter(conds, model, func=func, n_obs=n_inter, n_splits=n_splits, k=k)
    
    if not get_qm:
        return X, conds, inter_true, inter_gen, inter_conds
    
    kls = []
    rocs = []
    for i in range(inter_true.shape[0] // n_inter):
        
        true_sample = inter_true[i*n_inter : (i+1)*n_inter, :]#[:, 0].reshape((n_inter, 1))
        gen_sample = inter_gen[i*n_inter : (i+1)*n_inter, :]
        
        rocs.append(classify(np.concatenate((true_sample, gen_sample)), n_inter))
        kls.append(kl_divergence(true_sample, gen_sample))
    
    if rand_cond:
        res_dir = 'plots/rand_cond_inter_test'
    elif profiles:
        res_dir = 'plots/profiles'
    else:
        res_dir = 'plots/inter_test'
    
    current_time = time.strftime("%d.%m-%H:%M:%S")
    
    if func == 'parabolic':# or func == 'hyperbolic'
        file_name = f'{res_dir}_{current_time}_{func}_{k}_{n_samples}_{c_min}_{c_max}'
    else:
        file_name = f'{res_dir}_{current_time}_{func}_{n_samples}_{c_min}_{c_max}'
    
    if profiles:
        
        unique_inter_conds = len(np.unique(inter_conds))
        
        true_sample = inter_true.reshape(unique_inter_conds, n_inter, X.shape[-1])
        gen_sample  = inter_gen.reshape(unique_inter_conds, n_inter, X.shape[-1])
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        def animate(i):
            
            fig.clear()
            ax = fig.add_subplot(221)
            ax.set_xlim(min(conds), max(conds))
            ax.set_ylim(0, max(kls)+0.1)
            ax.set_title('Pairwise KL divergence')
            v1 = ax.vlines(conds, ymin=0, ymax=max(kls)+0.1, color='black', linewidth=2)
            s1 = ax.scatter(inter_conds[i], kls[i], zorder=10, color='orangered')
            s2 = ax.scatter(np.concatenate((inter_conds[:i], inter_conds[(i+1):])), np.concatenate((kls[:i], kls[(i+1):])), zorder=8, color='cornflowerblue')

            ax = fig.add_subplot(222)
            ax.set_xlim(min(conds), max(conds))
            ax.set_ylim(min(0.498, min(rocs)), 1+1e-3)
            ax.set_title('Pairwise classification ROC AUC')
            h1 = ax.hlines(0.5, xmin=min(conds), xmax=max(conds), colors='red', linewidth=2)
            v1 = ax.vlines(conds, ymin=min(0.498, min(rocs)), ymax=1+1e-3, color='black', linewidth=2)
            s1 = ax.scatter(inter_conds[i], rocs[i], zorder=10, color='orangered')
            s2 = ax.scatter(np.concatenate((inter_conds[:i], inter_conds[(i+1):])), np.concatenate((rocs[:i], rocs[(i+1):])), zorder=8, color='cornflowerblue')

            ax = fig.add_subplot(212)
            h1 = ax.hist(true_sample[i][:, 0], bins=n_inter//100, color='red', density=True, alpha=0.4)
            h2 = ax.hist(gen_sample[i][:, 0], bins=n_inter//100, color='blue', density=True, alpha=0.4)
            #create legend
            # handles = [Rectangle((0,0), 1, 1, color=c, ec="k") for c in ['orangered', 'cornflowerblue']]
            # labels  = ["True", "Generated"]
            # plt.legend(handles, labels)
            ax.set_title(f'Profiles for cond {i*max(conds)/((len(conds)-1)*100 - 1) + conds[0]:.2f}')
            ax.set_ylim(0, max(max(h1[0]), max(h2[0]))+0.05)
        
        ani = animation.FuncAnimation(fig, animate, interval=100, frames=range(unique_inter_conds))
        
        plt.close()
        
        ani.save(f'{file_name}.gif', writer='pillow')
        
        with open(f'{file_name}.gif', 'rb') as f:
            display(Image(data=f.read(), format='png'))
    
    else:
        
        fig = plt.figure(figsize=(12, 18))
        gs = gridspec.GridSpec(3, 2)
        
        # Plotting true data
        true_g = sns.jointplot(x=X[:, 1], y=X[:, 0], kind='scatter')
        _ = sfg(true_g, fig, gs[0, 0], 'True Data')
        
        # Plotting generated sample
        gen_g = sns.jointplot(x=X[:, 1], y=X_gen[:, 0], kind='scatter')
        _ = sfg(gen_g, fig, gs[0, 1], 'Generated Data')
        
        # Plotting inter true sample
        true_g = sns.jointplot(x=inter_true[:, 1], y=inter_true[:, 0], kind='scatter')
        _ = sfg(true_g, fig, gs[1, 0], 'True Interpolation')

        # Plotting inter gen sample
        gen_g = sns.jointplot(x=inter_gen[:, 1], y=inter_gen[:, 0], kind='scatter')
        _ = sfg(gen_g, fig, gs[1, 1], 'Generated Interpolation')

        # Calculating and plotting pairwise kl divergence between true inter samples and generated inter samples
        ax = fig.add_subplot(gs[2, 0])
        ax.vlines(conds, ymin=0, ymax=max(kls)+0.1, colors='black')
        ax.scatter(inter_conds, kls, zorder=10)
        ax.set_title('Pairwise KL divergence')

        # Pairwise ROC AUCs
        ax = fig.add_subplot(gs[2, 1])
        ax.vlines(conds, ymin=min(0.5, min(rocs)), ymax=1+1e-5, colors='black')
        ax.hlines(0.5, xmin=min(conds), xmax=max(conds), colors='red')
        ax.scatter(inter_conds, rocs, zorder=10)
        ax.set_title('Pairwise classification ROC AUC')

        gs.tight_layout(fig)
        plt.savefig(f'{file_name}.jpeg', format='jpeg')
        plt.show()
    
    return kls, rocs


def create_dir(func, opt, surrogate):
    
    exp_path = 'opt_res'
    
    if not os.path.isdir(exp_path): os.mkdir(exp_path)
    
    exp_path += f'/{func}'
    if not os.path.isdir(exp_path): os.mkdir(exp_path)
    
    exp_path += f'/{opt}'
    if not os.path.isdir(exp_path): os.mkdir(exp_path)
    
    exp_path += f'/{surrogate}'
    if not os.path.isdir(exp_path): os.mkdir(exp_path)
