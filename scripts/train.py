import wandb
import argparse
import time
import sys
import torch
from torch.utils.data import DataLoader
import hamiltorch
import os

from train_functions import FoMoH_gradient_descent, FoMoH_Kd_gradient_descent, make_data, get_time_stamp, backprop_routine

parser = argparse.ArgumentParser(description='train', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=0.0002)
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--device', type=str, default=None)
parser.add_argument('--model', type=str, choices=['logreg', 'cnn', 'cnn-cifar10'], default='logreg')
parser.add_argument('--optimizer', type=str, choices=['sgd', 'sgdn', 'adam'], default='sgd')
parser.add_argument('--method', type=str, choices=['FoMoH', 'FoMoH2d', 'FGD', 'BP', "FoMoH-BP", 'FoMoH3d'], default='FoMoH')
# parser.add_argument('--momentum', type=float, default=0.2)
parser.add_argument('--epsilon', type=float, default=1e-5)
parser.add_argument('--clip_mode', type=str, choices=['none'], default='none')
parser.add_argument('--clip_value', type=float, default=0.)
parser.add_argument('--weight_decay', type=float, default=0.)
parser.add_argument('--loss_min', type=float, default=0.)
parser.add_argument('--loss_max', type=float, default=20.)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--fwd_evals_per_iter', type=int, default=1)
parser.add_argument('--lr_sched', action='store_true')
parser.add_argument('--milestones', type=int, default=1000)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--no_wandb', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('--run_to_end', action='store_true')
parser.add_argument('--save-dir', type=str, default="./model/best_results/")
args = parser.parse_args()

if args.device is None and torch.cuda.is_available():
    args.device = torch.device('cuda')

if args.lr_sched: 
    save_path = args.save_dir + args.method + "_" + args.model + f"_seed_{args.seed}_sched.pt"
else:
    save_path = args.save_dir + args.method + "_" + args.model + f"_seed_{args.seed}.pt"
    
if args.save and os.path.exists(save_path):
    print("Save path exists already")
    sys.exit()
    
use_wandb = not args.no_wandb
    
if use_wandb:
    wandb.init(
        project='FoMoH', 
        name='{}_{}'.format(args.method, get_time_stamp()), 
        config=args)
    config = wandb.config
    print('Arguments:\n{}\n'.format(' '.join(sys.argv[1:])))
    print('Config:')
    for k, v in config.items():
        print('{}: {}'.format(k, v))
    print()
    if config.device.startswith('cuda'):
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
else:
    config = args

hamiltorch.set_random_seed(config.seed)
device = torch.device(config.device)
time_start = time.time()

dataset_train, dataset_valid = make_data(config.model)
train_loader = DataLoader(dataset_train, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
val_loader = DataLoader(dataset_valid, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=True)

if config.method == 'FoMoH' or config.method == 'FGD' or config.method == 'FoMoH-BP':
    if config.method == 'FoMoH' or config.method == 'FoMoH-BP':
        hess_norm = True
    else:
        hess_norm = False
    if config.clip_mode == "none":
        clip_value = 0.0
    else:
        clip_value = config.clip_value
        
    if config.method == 'FoMoH-BP':
        backprop = True
    else:
        backprop = False
        
    save_dict = FoMoH_gradient_descent(config.model, epochs=config.epochs, n_sample_directions=config.fwd_evals_per_iter, lr=config.lr, clip_value=clip_value, hess=hess_norm, weight_decay=config.weight_decay, optimizer=config.optimizer, train_loader=train_loader, val_loader=val_loader, epsilon=config.epsilon, device = device, use_wandb = use_wandb, backprop = backprop, save=config.save, run_to_end = config.run_to_end, lr_sched = config.lr_sched, milestones = config.milestones, gamma = config.gamma)

elif config.method == 'FoMoH2d' or config.method == 'FoMoH3d':
    if config.clip_mode == "none":
        clip_value = 0.0
    else:
        clip_value = config.clip_value

    if config.method == 'FoMoH2d':
        K = 2
    if config.method == 'FoMoH3d':
        K = 3
        
    save_dict = FoMoH_Kd_gradient_descent(config.model, epochs=config.epochs, n_sample_directions=config.fwd_evals_per_iter, lr=config.lr, clip_value=clip_value, train_loader=train_loader, val_loader=val_loader, device = device, use_wandb = use_wandb, save=config.save, run_to_end = config.run_to_end, K=K, lr_sched = config.lr_sched, milestones = config.milestones, gamma = config.gamma)
    
elif config.method == 'BP':
    if config.clip_mode == "none":
        clip_value = 0.0
    else:
        clip_value = config.clip_value
    save_dict = backprop_routine(config.model, epochs=config.epochs, lr=config.lr, clip_value=clip_value, weight_decay=config.weight_decay, optimizer=config.optimizer, train_loader=train_loader, val_loader=val_loader, device = device, use_wandb = use_wandb, save=config.save, run_to_end = config.run_to_end, lr_sched = config.lr_sched, milestones = config.milestones, gamma = config.gamma)
    
else:
    raise RuntimeError('Unknown method: {}'.format(config.method))

if use_wandb:
    wandb.finish()
if config.save:
    save_dict["args"] = config
    torch.save(save_dict, save_path)
    


# wandb sweep XXX.yml
# wandb agent sweep_id --count 100 

# If needed to kill wandb:
# ps aux|grep wandb|grep -v grep | awk '{print $2}'|xargs kill -9
