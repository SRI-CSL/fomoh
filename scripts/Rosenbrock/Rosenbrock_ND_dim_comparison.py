from fomoh.hyperdual import HyperTensor as htorch
from fomoh.opt import plane_step_2d, plane_step_Nd, invert_matrix
import torch
import hamiltorch
from tqdm import tqdm
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

def rosenbrock_ND(x):
    term1 = (1 - x[:, :-1])**2
    term2 = 100 * (x[:, 1:] - x[:, :-1]**2)**2
    return (term1 + term2).sum(-1)

def rosenbrock_jacobian_ND(x):
    # Compute basic derivatives
    diff = x[:, 1:] - x[:, :-1]**2
    d = -2 * (1 - x[:, :-1]) - 400 * x[:, :-1] * diff
    d_next = 200 * diff

    # Initialize Jacobian to zero
    jacobian = torch.zeros_like(x)

    # Set derivatives, taking care of boundary conditions
    jacobian[:, :-1] += d
    jacobian[:, 1:] += d_next
    return jacobian

def rosenbrock_hessian_ND(x):
    return torch.autograd.functional.hessian(rosenbrock_ND, x).squeeze()
    
def forward_plane_search(x, D, fun, T, lr):
    loss_plane = []
    for t in range(T):
        try:
            v = plane_step_Nd(fun, x, N=D)
        except:
            print(f"Exiting at iteration {t} of {T} due to linalgerror")
            return loss_plane
        x = x + lr*v
        loss_plane.append(fun(x).item())
    return loss_plane

def newton_opt(x, fun, T, lr=1.0):
    loss_newt = []
    for t in range(T):
        v = - (invert_matrix(rosenbrock_hessian_ND(x)) @ rosenbrock_jacobian_ND(x).t()).t()
        x = x + lr*v
        loss_newt.append(fun(x).item())
    return loss_newt

def plot_mean_std_fill_multiple(data_groups, labels, colors, fs=12, D = 2):
    """
    Plots the mean and standard deviation as a filled area around the mean line
    for an arbitrary number of data groups, each being a list of lists.
    
    :param data_groups: A list where each element is a data group (list of lists) to plot.
    :param labels: A list of labels for each data group. Must match the length of data_groups.
    """
    if len(data_groups) != len(labels):
        raise ValueError("Each data group must have a corresponding label.")
    
    i = 0
    for data_group, label, c in zip(data_groups, labels, colors):
        # Flatten the data group to a single list for mean and std calculations
        all_values = []
        for data in data_group:
            if data[0].__class__ is htorch:
                values = torch.tensor([-l.real.item() for l in data])
            else:
                values = torch.tensor([l for l in data])
            all_values.append(values)
        all_values = torch.stack(all_values)
        mean = all_values.mean(0).numpy()
        
        
        plt.plot(all_values.numpy().T, alpha = 0.2, color = c)
        plt.plot(mean, label=f"{label}", color = c, linewidth = 2)
        
        i += 1
    plt.xlabel("Iteration", fontsize = fs)
    plt.ylabel(f" {D}D Rosenbrock Function", fontsize = fs)
    plt.legend(fontsize = fs,loc='upper center', bbox_to_anchor=(0.47, 0.5))
    plt.yscale("log")
    plt.xscale("log")
    # plt.tight_layout()
    plt.grid()
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Rosenbrock ND Comparison', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dim-obj', type=int, default=2)
    parser.add_argument('--newton', action='store_true')
    parser.add_argument('--save-file', type=str, default='./plots/rosenbrock_plane_comparison.pt')

    args = parser.parse_args()


    hamiltorch.set_random_seed(4)

    T = args.epochs
    sigma = 1.0
    S = 10

    log_prob_fun = lambda x, v : -rosenbrock_ND(htorch(x,v,v))#.log()

    lp_full_list = []
    lp_newton_full_list = []
    d_list = []
    for d in tqdm(range(2, args.dim_obj + 1)):
        lp_list = []
        for s in range(S):
            hamiltorch.set_random_seed(s)
            sample_init = torch.randn(args.dim_obj).view(1,args.dim_obj)

            #FoMoH-KD
            x = sample_init.clone()
            loss_plane = forward_plane_search(x, D = d, fun = rosenbrock_ND, T=T, lr = args.lr)
            lp_list.append(loss_plane)

            if args.newton and d == 2:
                lp_newton_full_list.append(newton_opt(x, fun = rosenbrock_ND, T=T, lr=1.0))
        
        d_list.append(d)
        lp_full_list.append(lp_list)

        save_dict = {"FoMoH-KD": lp_full_list, "D_list":d_list, "Newton": lp_newton_full_list, "Args": args}
        torch.save(save_dict, args.save_file)
    save_dict = {"FoMoH-KD": lp_full_list, "D_list":d_list, "Newton": lp_newton_full_list, "Args": args}
    torch.save(save_dict, args.save_file)


if __name__ == "__main__":
    main()
    
    
    # python Rosenbrock_ND_dim_comparison.py --epochs 100 --dim-obj 10 --save-file ./plots/rosenbrock_comparison_plane_dim_comparison_10D.pt --newton