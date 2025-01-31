# Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks on Distributed Heterogeneous Data
This hub stores the code for paper *Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks on Distributed Heterogeneous Data*.

For the tutorial related to the paper *Mean Aggregator Is More Robust Than Robust Aggregators Under Label Poisoning Attacks*, please refer to the `README_IJCAI.md` file.
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
- `main CMomentum(-xxx).py`, : program entry
  * `main CMomentum.py`: compute classification accuracies of different aggregators (Fig. 1, 2, 4, 5)
  * `main CMomentum-hetero-bound.py`: compute heterogeneity of regular gradients and disturbances of poisoned gradients (Fig. 3, 6)
  * `main CMomentum-A-xi-NN.py`: compute classification accuracies of different aggregators under different data distributions and attack strengths (Fig. 7)
  * `main CMomentum-variance.py`: compute variance of regular and poisoned stochastic gradients (Fig. 8)
-  `draw_fig`: directories containing the codes that draw the figures in paper


## Runing
### Run CMomentum
```bash
python "main CMomentum.py"  --aggregation <aggregation-name> --attack <attack-name> --data-partition <data-partition>
# ========================
# e.g.
# python "main CMomentum.py" --aggregation trimmed-mean --attack label_flipping --data-partition noniid
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

python draw-MultiFig-Momentum.py 

python draw_A_xi_Momentum.py

python draw_alpha_prob_Momentum.py
```
