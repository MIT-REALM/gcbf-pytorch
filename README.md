# GCBF-PyTorch

[![Conference](https://img.shields.io/badge/CoRL-Accepted-success)](https://mit-realm.github.io/gcbf-website/)

PyTorch Official Implementation of CoRL 2023 Paper: [S Zhang](https://syzhang092218-source.github.io), [K Garg](https://kunalgarg.mit.edu/), [C Fan](https://chuchu.mit.edu): "[Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control](https://mit-realm.github.io/gcbf-website/)"

!!!!!!!!!!
**We have improved GCBF to [GCBF+](https://mit-realm.github.io/gcbfplus-website/)!! Check out the new code [here](https://github.com/MIT-REALM/gcbfplus).**
!!!!!!!!!!

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

- `--env`: environment, including `SimpleCar`, `SimpleDrone`, `DubinsCar`
- `--algo`: algorithm, including `gcbf`, `macbf`
- `-n`: number of agents
- `--steps`: number of training steps
- `--batch-size`: batch size
- `--area-size`: side length of the environment
- `--obs`: number of obstacles
- `--seed`: random seed
- `--gpu`: GPU ID
- `--cpu`: use CPU
- `--log-path`: path to save the training logs

By default, the training uses the hyperparameters in [`hyperparams.yaml`](gcbf/trainer/hyperparams.yaml). To use different hyperparameters, one can use the flag `--cus` and then use the flags `--h-dot-coef` and `--action-coef` to specify the new hyper-parameters. 

### Test

To test the learned model, use:

```bash
python test.py --path <path-to-log> --epi <number-of-episodes>
```

This should report the safety rate, goal reaching rate, and success rate of the learned model, and generate videos of the learned model in `<path-to-log>/videos`. Use the following flags to customize the test:

- `-n`: number of agents
- `--obs`: number of obstacles
- `--area-size`: side length of the environment
- `--sense-radius`: sensing radius of the agents
- `--iter`: number of training iterations of the model
- `--epi`: number of episodes to test
- `--seed`: random seed
- `--gpu`: GPU ID
- `--cpu`: use CPU
- `--no-video`: do not generate videos

There is also a nominal controller implemented for each environment for goal reaching. They can be tested using:

```bash
python test.py --env <env> -n <number-of-agents> --epi <number-of-episodes>
```

### Pre-trained models
We provide the pre-trained models in the folder [`./pretrained`](pretrained).

## Citation

```
@inproceedings{zhang2023gcbf,
      title={Neural Graph Control Barrier Functions Guided Distributed Collision-avoidance Multi-agent Control},
      author={Zhang, Songyuan and Garg, Kunal and Fan, Chuchu},
      booktitle={7th Annual Conference on Robot Learning},
      year={2023}
}
```

## Acknowledgement

The developers were partially supported by MITRE during the project.
