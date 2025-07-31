from argsParser import args

from ByrdLab import FEATURE_TYPE
from ByrdLab.aggregation import C_mean, C_trimmed_mean, C_faba, C_centered_clipping, C_LFighter
from ByrdLab.attack import C_gaussian, C_same_value, C_sign_flipping, feature_label_random, \
                            label_flipping, label_random, furthest_label_flipping, adversarial_label_flipping, feature_label_random
from ByrdLab.centraliedAlgorithm import CSGD, CSGD_under_DPA, CMomentum_under_DPA
from ByrdLab.library.cache_io import dump_file_in_cache, load_file_in_cache
from ByrdLab.library.dataset import ijcnn, mnist, fashionmnist, cifar10, mnist_sorted_by_labels
from ByrdLab.library.learnRateController import ladder_lr, one_over_sqrt_k_lr
from ByrdLab.library.partition import (LabelSeperation, TrivalPartition,
                                   iidPartition, DirichletIiiPartition, DirichletMildPartition, DirichletNoniidPartition,
                                    DirichletPartition_a, DirichletPartition_b, DirichletPartition_c, DirichletPartition_d, DirichletPartition_e, DirichletPartition_f)
from ByrdLab.library.tool import log
from ByrdLab.tasks.logisticRegression import LogisticRegressionTask
from ByrdLab.tasks.softmaxRegression import softmaxRegressionTask
from ByrdLab.tasks.leastSquare import LeastSquareToySet, LeastSquareToyTask
from ByrdLab.tasks.neuralNetwork import NeuralNetworkTask

node_size = 10
byzantine_size = 1

all_nodes = list(range(node_size))
honest_nodes = list(range(node_size - byzantine_size))
byzantine_nodes = [node for node in all_nodes if node not in honest_nodes]

args.lr_ctrl = 'constant'
# run for centralized algorithm
# ===========================================

# -------------------------------------------
# define learning task
# -------------------------------------------
data_package = mnist()
task = softmaxRegressionTask(data_package, batch_size=32)

# data_package = cifar10()
# task = NeuralNetworkTask(data_package, batch_size=32)

# data_package = mnist()
# task = NeuralNetworkTask(data_package, batch_size=32)
# ===========================================

# -------------------------------------------
# define attack
# -------------------------------------------
if args.attack == 'none':
    attack = None
elif args.attack == 'label_flipping':
    attack = label_flipping()
elif args.attack == 'label_random':
    attack = label_random()
elif args.attack == 'feature_label_random':
    attack = feature_label_random()
elif args.attack == 'furthest_label_flipping':
    attack = furthest_label_flipping()
if args.attack == 'none':
    attack_name = 'baseline'
    byzantine_size = 0

    honest_nodes = list(range(node_size))
    byzantine_nodes = []
else:
    attack_name = attack.name


# -------------------------------------------
# define learning rate control rule
# -------------------------------------------
if args.lr_ctrl == 'constant':
    lr_ctrl = None
elif args.lr_ctrl == '1/sqrt k':
    lr_ctrl = one_over_sqrt_k_lr(a=1, b=1)
    # super_params = task.super_params
    # total_iterations = super_params['rounds']*super_params['display_interval']
    # lr_ctrl = one_over_sqrt_k_lr(total_iteration=total_iterations,
    #                              a=math.sqrt(1001), b=1000)
elif args.lr_ctrl == 'ladder':
    decreasing_iter_ls = [5000, 10000, 15000]
    proportion_ls = [0.3, 0.2, 0.1]
    lr_ctrl = ladder_lr(decreasing_iter_ls, proportion_ls)
else:
    assert False, 'unknown lr-ctrl'

# ===========================================
    
    
# -------------------------------------------
# define data partition
# -------------------------------------------
if args.data_partition == 'trival':
    partition_cls = TrivalPartition
elif args.data_partition == 'iid':
    partition_cls = iidPartition
elif args.data_partition == 'noniid':
    partition_cls = LabelSeperation
elif args.data_partition == 'dirichlet_mild':
    partition_cls = DirichletMildPartition
else:
    assert False, 'unknown data-partition'
# ===========================================
    

# -------------------------------------------
# define aggregation
# -------------------------------------------
if args.aggregation == 'mean':
    aggregation = C_mean(honest_nodes, byzantine_nodes)
elif args.aggregation == 'trimmed-mean':
    aggregation = C_trimmed_mean(honest_nodes, byzantine_nodes)
elif args.aggregation == 'faba':
    aggregation = C_faba(honest_nodes, byzantine_nodes)
elif args.aggregation == 'cc':
    if args.data_partition == 'iid':
        threshold = 0.1
    else:
        threshold = 0.3
    aggregation = C_centered_clipping(honest_nodes, byzantine_nodes, threshold=threshold)
elif args.aggregation == 'lfighter':
    aggregation = C_LFighter(honest_nodes, byzantine_nodes)
else:
    assert False, 'unknown aggregation'

# ===========================================

workspace = []
mark_on_title = ''
fix_seed = not args.no_fixed_seed
seed = args.seed
record_in_file = not args.without_record
step_agg = args.step_agg

# initilize optimizer
if 'label' in attack_name:
  if args.aggregation == 'lfighter':
      env = CSGD_with_LFighter_under_DPA(aggregation=aggregation, honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, attack=attack, step_agg = step_agg,
             weight_decay=task.weight_decay, data_package=task.data_package,
             model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
             initialize_fn=task.initialize_fn,
             get_train_iter=task.get_train_iter,
             get_test_iter=task.get_test_iter,
             partition_cls=partition_cls, lr_ctrl=lr_ctrl,
             fix_seed=fix_seed, seed=seed,
             **task.super_params)
  else:
    env = CSGD_under_DPA(aggregation=aggregation, honest_nodes=honest_nodes, byzantine_nodes=byzantine_nodes, attack=attack, step_agg = step_agg,
             weight_decay=task.weight_decay, data_package=task.data_package,
             model=task.model, loss_fn=task.loss_fn, test_fn=task.test_fn,
             initialize_fn=task.initialize_fn,
             get_train_iter=task.get_train_iter,
             get_test_iter=task.get_test_iter,
             partition_cls=partition_cls, lr_ctrl=lr_ctrl,
             fix_seed=fix_seed, seed=seed,
             **task.super_params)

title = '{}_{}_{}'.format(env.name, attack_name, aggregation.name)

if lr_ctrl != None:
    title = title + '_' + lr_ctrl.name
if mark_on_title != '':
    title = title + '_' + mark_on_title

data_package = task.data_package
super_params = task.super_params

# print the running information
print('=========================================================')
print('[Task] ' + task.name + ': ' + title)
print('=========================================================')
print('[Setting]')
print('{:12s} model={}'.format('[task]', task.model_name))
print('{:12s} dataset={} partition={}'.format(
    '[dataset]', data_package.name, env.partition_name))
print('{:12s} name={} aggregation={} attack={}'.format(
    '[Algorithm]', env.name, aggregation.name, attack_name))
print('{:12s} lr={} lr_ctrl={}, weight_decay={}'.format(
    '[Optimizer]', super_params['lr'], env.lr_ctrl.name, task.weight_decay))
print('{:12s} honest_size={}, byzantine_size={}'.format(
    '[Graph]', node_size - byzantine_size, byzantine_size))
print('{:12s} rounds={}, display_interval={}, total iterations={}'.format(
    '[Running]', env.rounds, env.display_interval, env.total_iterations))
print('{:12s} seed={}, fix_seed={}'.format('[Randomness]', seed, fix_seed))
print('{:12s} record_in_file={}'.format('[System]', record_in_file))
print('-------------------------------------------')

log('[Start Running]')
_, loss_path, acc_path = env.run()


record = {
    'dataset': data_package.name,
    'dataset_size': len(data_package.train_set),
    'dataset_feature_dimension': data_package.feature_dimension,
    'lr': super_params['lr'],
    'weight_decay': task.weight_decay,
    'honest_size': node_size - byzantine_size,
    'byzantine_size': byzantine_size,
    'rounds': env.rounds,
    'display_interval': env.display_interval,
    'total_iterations': env.total_iterations,
    'loss_path': loss_path,
    'acc_path': acc_path,
    'fix_seed': fix_seed,
    'seed': seed
}

if record_in_file:
    path_list = [task.name, f'Centralized_n={node_size}_b={byzantine_size}', env.partition_name] + workspace
    dump_file_in_cache(title, record, path_list=path_list)
print('-------------------------------------------')

