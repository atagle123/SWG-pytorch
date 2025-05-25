# Self Weighted Guidance Repository

Welcome to the Self Weighted Guidance repository in PyTorch. 


## Installation

We recommend using [Miniconda](https://docs.conda.io/en/latest/miniconda.html) to manage dependencies.


### 1. Clone the Repository

```bash
git clone https://github.com/atagle123/SWG-pytorch
cd swg-pt
```

### 2. Create and Activate a Conda Environment

```bash
conda env create -f environment.yml
conda activate swg_pytorch
conda develop .
```

## Usability

To modify the main hyperparameters for experiments, edit the following file:

```bash
configs/D4RL/config.yaml
```

## Training

Change dataset_name with the dataset to train

```bash
python scripts/train.py method=swg datasets={dataset_name} sseed={your_seed}
```

## Evaluating

Evaluate models

```bash
python scripts/evaluate.py
```