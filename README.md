# GCBF-PyTorch

[![Conference](https://img.shields.io/badge/CoRL-Accepted-success)](https://mit-realm.github.io/gcbf-website/)

PyTorch Official Implementation of CoRL 2023 Paper: [S Zhang](https://syzhang092218-source.github.io), [K Garg](https://kunalgarg.mit.edu/), [C Fan](https://chuchu.mit.edu): "[Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control](https://mit-realm.github.io/gcbf-website/)"

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n gcbf python=3.9
conda activate gcbf
pip install -r requirements.txt
```

Then you need to install additional packages for `torch_geometric` including `pyg_lib`, `torch_scatter`, `torch_sparse`, `torch_cluster`, `torch_spline_conv` following the [official website](https://pytorch-geometric.readthedocs.io/en/latest/).

## Installation

Install GCBF: 

```bash
pip install -e .
```

## Run

### Environments

We provide 3 environments including `SimpleCar`, `SimpleDrone`, and `DubinsCar`. 

### Hyper-parameters

To reproduce the results shown in our paper, one can refer to [`hyperparams.yaml`](gcbf/trainer/hyperparams.yaml).

### Train

To train the model, use:

```bash
python train.py --algo gcbf --env DubinsCar -n 16 --steps 500000
```

In our paper, we use 16 agents with 500000 training steps. The training logs will be saved in folder `./logs/<env>/<algo>/seed<seed>_<training-start-time>`. We also provide the following flags:

- `--seed`: random seed
- `--algo`: algorithm, including `gcbf`, `macbf`

## Test

To test the learned model in the same environment as that in training, use:

```bash
python test.py --path <path-to-log> --epi <number-of-episodes>
```

To test the learned model in an environment with different number of agents, use:

```bash
python test.py --path <path-to-log> --epi <number-of-episodes> -n <number-of-agents>
```

You can add the flag `--no-vodeo` for faster tests without generating videos.