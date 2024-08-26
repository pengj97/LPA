# Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks
This hub stores the code for paper *Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks*.

## Install
1. Download the dependant packages (c.f. `install.sh`):
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
- `main CSGD(-xxx).py` `main CMomentum(-xxx).py`, : program entry
  * `main CSGD.py` `main CSGD-LFighter` `main CMomentum.py` `main CMomentum-LFighter.py`: compute classification accuracies of different aggregators
  * `main CSGD-`
-  `draw_fig`: directories containing the codes that draw the figures in paper


## Runing
### Run CSGD
```bash
python "main CSGD.py"  --aggregation <aggregation-name> --attack <attack-name> --data-partition <data-partition>
# ========================
# e.g.
# python "main CSGD.py" --aggregation trimmed-mean --attack label_flipping --data-partition noniid
```

### Run CSGD-LFighter
```bash
python "main CSGD-LFighter.py"   --attack <attack-name> --data-partition <data-partition>
# ========================
# e.g.
# python "main CSGD-LFighter.py" --attack label_flipping --data-partition noniid
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
