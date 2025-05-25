# Self Weighted Guidance Repository

Welcome to the Self Weighted Guidance repository for toy examples in PyTorch. 


## Installation

We recommend using [Miniconda] to manage dependencies.


### 1. Clone the Repository

```bash
git clone https://github.com/atagle123/SWG-pytorch
cd swg-pt
```

### 2. Create and Activate a Conda Environment

```bash
conda env create -f environment.yml
conda activate swg_pytorch_toy
conda develop .
```

## Usability

To modify the main hyperparameters for experiments, edit the following file:

```bash
configs/toy/config.yaml
```

## Training

Change dataset_name with the dataset to train

```bash
python scripts/train.py datasets={dataset_name}
```

## Evaluating

Get the plots for the main paper 

```bash
python scripts/get_diffusion_plots.py
```