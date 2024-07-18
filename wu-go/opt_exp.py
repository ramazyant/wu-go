import os
import gc
import sys
import ego
import time
import pickle
import ego_models
from utils import create_dir
from test_functions import *


def test(model='GAN', func=three_hump_camel(), opt='WU-GO'):
    
    seeds = [2, 3, 5, 7, 11, 13, 17, 19, 23, 73] # seeds control the location of initial responses given to the model
    
    for seed in seeds:
        
        # running a single test...
        
        if model not in ['GP', 'DGP']:
            model_ = ego_models.EGO_model(model=model, n_inputs=func.dim)
        else:
            model_ = model
            
        opt_model = ego.EGO(model=model_, func=func, opt=opt, seed=seed)
        
        res = opt_model.optimise()
    
        # saving results...

        create_dir(func.name, opt, opt_model.model.name)

        with open(f'opt_res/{func.name}/{opt}/{opt_model.model.name}/{time.strftime("%d.%m-%H:%M:%S")}.pkl', 'wb') as f:
            pickle.dump(res, f)
        
        del res
        gc.collect()


def main():
    
    # the following code cell will run all experiments we present in our paper and save them in the 'optim_res' directory
    func  = None
    opt   = sys.argv[2]
    model = sys.argv[3]
    
    for f in [three_hump_camel(), ackley(), levi(), himmelblau(), rosenbrock(dim=8), rosenbrock(), tang()]:
        if f.name == sys.argv[1]:
            func = f
            break

    test(model, func, opt)


if __name__ == '__main__':
    main()
