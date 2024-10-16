# Global Optimisation of Black-Box Functions with Generative Models in the Wasserstein Space

Official code for the paper [Global Optimisation of Black-Box Functions with Generative Models in the Wasserstein Space](https://arxiv.org/abs/2407.11917). 

Tigran Ramazyan*, Mikhail Hushchyn, Denis Derkach.

## Pypi package

[WAGGON: WAsserstein Global Gradient-free OptimisatioN](https://github.com/hse-cs/waggon/tree/main)

## Installation

- Create conda environment:

```sh
conda create -n wugo python=3.10
conda activate wugo
```

- Install core dependencies:

```sh
pip install -r requirements.txt
```

## Experiments

- To run experiments:
```bash
    python opt_exp.py experiment acquisition_function surrogate_model
```

Available options:
- experiment: `three_hump_camel`, `ackley`, `levi`, `himmelblau`, `rosenbrock8`, `rosenbrock20`, `tang`
- acquisition_function: `WU-GO`, `EI`, `LCB`
- surrogate_model: `GAN`, `BNN`, `DE`, `GP`, `DGP`


## Citation

```
@misc{ramazyan2024globaloptimisationblackboxfunctions,
      title={Global Optimisation of Black-Box Functions with Generative Models in the Wasserstein Space}, 
      author={Tigran Ramazyan and Mikhail Hushchyn and Denis Derkach},
      year={2024},
      eprint={2407.11917},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.11917}, 
}
```
