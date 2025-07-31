# Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks
This README document is for our paper *Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks* published in IJCAI 2024.

## Install
1. Download the dependant packages:
- python 3.8.10
- pytorch 1.9.0
- matplotlib 3.3.4
- networkx 2.5.1

2. Download the dataset to the directory `./dataset` and create a directory named `./record`. The experiment outputs will be stored in `./record`.

- *MNIST*: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- *CIFAR10*: [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

## Construction
The main programs can be found in the following files:
- `ByrdLab`: main codes
- `main CSGD.py` `main CSGD-A-xi-NN.py`, : program entry
-  `draw_fig`: directories containing the codes that draw the figures in paper


## Runing
### Run CSGD
```bash
python "main CSGD.py"  --aggregation <aggregation-name> --attack <attack-name> --data-partition <data-partition>
# ========================
# e.g.
# python "main CSGD.py" --aggregation trimmed-mean --attack label_flipping --data-partition noniid
```

> The arguments can be
>
>
> `<aggregation-name>`: 
> - mean
> - trimmed-mean
> - faba
> - cc
> - lfighter
>
> `<attack-name>`: 
> - label_flipping (which executes static label flipping attacks)
> - furthest_label_flipping (which executes dynamic label flipping attacks)

>
> `<data-partition>`: 
> - iid
> - dirichlet_mild
> - noniid

---


# ====================
# Fig
```
cd draw_fig

python draw.py 

python draw_A_xi.py

python draw_alpha_prob.py
```
