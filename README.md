# Sequential-Neural-Information-Diffusion-Model-with-Structure-Attention

This repository provides reference codes as described in the paper:

>**A Sequential Neural Information Diffusion Model with Structure Attention.**  
>**Zhitao Wang, Chengyao Chen and Wenjie Li.**  
>**CIKM, 2018.** 

## Environment
The code is implemented with Tensorflow. Requirments:  
&emsp;1. Python 2.7  
&emsp;2. Numpy  
&emsp;3. Tensorflow  
&emsp;4. tqdm (for training process display)   

## Run
Defalut:
    python train.py
optional arguments:
  -h, --help            show this help message and exit  
  -l LR, --lr LR        learning rate  
  -x XDIM, --xdim XDIM  embedding dimension  
  -e HDIM, --hdim HDIM  hidden dimension  
  -d DATA, --data DATA  data name  
  -g GPU, --gpu GPU     gpu id  
  -b BS, --bs BS        batch size  
  -f FREQ, --freq FREQ  validation frequency  
  -n NEPOCH, --nepoch NEPOCH number of training epochs

## Citing
    @inproceedings{Wang:2018:SNI:3269206.3269275,
    author = {Wang, Zhitao and Chen, Chengyao and LI, Wenjie},
    title = {A Sequential Neural Information Diffusion Model with Structure Attention},
    booktitle = {Proceedings of the 27th ACM International Conference on Information and Knowledge Management},
    series = {CIKM '18},
    year = {2018},
    location = {Torino, Italy},
    pages = {1795--1798}
    } 


  
