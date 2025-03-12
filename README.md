# rl-for-adaptive-testing

## Setup
Make sure you have miniforge or use conda with conda forge
```bash
conda env create -f environment.yml
```

Setting up conda with pytorch gpu
```bash
conda env create -n cat_cuda python=3.12
```
```bash
conda activate cat_cuda
```
```bash
conda install cudatoolkit -c anaconda -y
```
```bash
conda install pytorch-cuda=12.1 -c pytorch -c nvidia -y
```
```bash
conda install pytorch torchvision torchaudio -c pytorch -c nvidia - y
```
```bash
conda env update -f environment_cuda.yml --prune
```


## Computer Adaptive Testing Environment
Env code is located in `cat_env.py`. You can run the file to see example use.