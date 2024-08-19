from fomoh.hyperdual import HyperTensor as htorch
from fomoh.opt import plane_step_2d, plane_step_Nd, invert_matrix, softabs
import torch
import hamiltorch
from tqdm.notebook import tqdm
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

def rosenbrock_hessian_ND(x, alpha = 0.0, softabs_flag = False):
    if softabs_flag:
        return softabs(torch.autograd.functional.hessian(rosenbrock_ND, x).squeeze()) + alpha * torch.eye(x.shape[1])
    else:
        return torch.autograd.functional.hessian(rosenbrock_ND, x).squeeze() + alpha * torch.eye(x.shape[1])


def get_directional_derivative(out, param_directions, hess, epsilon = 10e-7):
    if hess:
        directional_derivative = (out.eps1 * param_directions.t() / abs(out.eps1eps2)).mean(-1)
    else:
        directional_derivative = (out.eps1 * param_directions.t()).mean(-1)
    return directional_derivative

def single_sample_step(theta, log_prob_fun, log_prob = None, tangent = None, sigma = 1.0, clip_value = None, hess_norm=True, eps = 10e-7, reduce = "sum"):
    
    if len(theta.shape) > 1:
        B = theta.real.shape[0]
    else:
        B = 1
    
    if log_prob is None or tangent is None:
        tangent = torch.randn_like(theta)
        log_prob = log_prob_fun(theta, tangent)
    
    # Proposal:
    directional_derivative = get_directional_derivative(log_prob, tangent, hess_norm, eps)

    # Clip  gradients
    if clip_value is not None:
        directional_derivative = torch.clamp(directional_derivative, -abs(clip_value), abs(clip_value))

    theta_prop = theta + sigma * directional_derivative #+ 3 * tangent

    # tangent for the next step if accepted: Assume it is faster to run it every step
    tangent = torch.randn_like(theta)
    log_prob_prop = log_prob_fun(theta_prop, tangent)

    return theta_prop, 1, log_prob_prop, tangent


def fomoh_sampler(log_prob_fun, T, sample_init, sigma, clip_value = None, hess_norm = True):
    sample = sample_init.clone()
    accepts = []
    samples = []
    iteration = []
    lp = []
    tepoch = tqdm(range(T))
    for i in tepoch:
        if i == 0:
            sample, acc, log_prob, tangent = single_sample_step(sample, log_prob_fun, log_prob = None, tangent = None, sigma = sigma, clip_value = clip_value, hess_norm=hess_norm, eps = 10e-7)
        else:
            sample, acc, log_prob, tangent = single_sample_step(sample, log_prob_fun, log_prob = log_prob, tangent = tangent, sigma = sigma, clip_value = clip_value, hess_norm=hess_norm, eps = 10e-7)
        accepts.append(acc)
        lp.append(log_prob)
        samples.append(sample.clone().cpu())
    samples = torch.stack(samples)
    return samples, lp, iteration, accepts

    
def forward_plane_search(x, D, fun, T, alpha, softabs_flag = False, clip_value=0.0):
    loss_plane = []
    for t in range(T):
        v = plane_step_Nd(fun, x, N=D, alpha = alpha, softabs_flag = softabs_flag)
        if clip_value != 0.0:
            v = torch.clamp(v, -abs(clip_value), abs(clip_value))
        x = x + v
        loss_plane.append(fun(x).item())
    return loss_plane


def torch_optim(sample_init, lr, T, optim = torch.optim.Adam, fun = rosenbrock_ND):
    # Initialize variable x as a 2D tensor. Here, .requires_grad_() indicates that we want to optimize it.
    x = sample_init.clone().requires_grad_()

    # Choose the Adam optimizer
    optimizer = optim([x], lr=lr)
    
    loss_list = []
    # Run the optimization loop
    for step in range(T):
        optimizer.zero_grad()   # Zero the gradients
        loss = fun(x)    # Compute the Rosenbrock function
        loss.backward()         # Backpropagate to compute gradients
        optimizer.step()        # Update the variables
        loss_list.append(loss.item())
    return loss_list

def newton_opt(x, fun, T, lr=1.0, alpha = 0.0, softabs_flag=False, clip_value=0.0):
    loss_newt = []
    for t in range(T):
        v = - (invert_matrix(rosenbrock_hessian_ND(x, alpha, softabs_flag) ) @ rosenbrock_jacobian_ND(x).t()).t()
        if clip_value != 0.0:
            v = torch.clamp(v, -abs(clip_value), abs(clip_value))
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
    # plt.title("Optimization of the Rosenblock Function", fontsize = fs)
    # Adjust layout to make room for the legend below the plot
    # plt.subplots_adjust(bottom=0.2)
    # Place the legend below the plot in two columns
    # plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), shadow=False, ncol=2, fontsize = fs)
    plt.legend(fontsize = fs,loc='upper center', bbox_to_anchor=(0.47, 0.5))
    plt.yscale("log")
    plt.xscale("log")
    # plt.tight_layout()
    plt.grid()
    # plt.show()

def main():
    parser = argparse.ArgumentParser(description='Rosenbrock ND Comparison', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-FoMoH', type=float, default=1.)
    parser.add_argument('--lr-adam', type=float, default=0.2)
    parser.add_argument('--clip-value', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dim-obj', type=int, default=2)
    parser.add_argument('--plane-dim', type=int, default=2)
    parser.add_argument('--save-file', type=str, default='./plots/rosenbrock_comparison.pt')
    parser.add_argument('--softabs', action='store_true')

    args = parser.parse_args()


    hamiltorch.set_random_seed(4)

    T = args.epochs
    sigma = 1.0
    S = 10
    lr = args.lr

    log_prob_fun = lambda x, v : -rosenbrock_ND(htorch(x,v,v))#.log()

    lp_hess_norm_list = []
    lp_hess_plane = []
    lp_grad_list_lr = []
    lp_grad_list_2lr = []
    lp_adam = []
    lp_newton = []

    for s in range(S):
        hamiltorch.set_random_seed(s)
        sample_init = torch.randn(args.dim_obj).view(1,args.dim_obj)

        #FoMoH
        samples_hn, lp_hn, iteration_hn, accepts_hn = fomoh_sampler(log_prob_fun, T, sample_init, args.lr_FoMoH)
        lp_hess_norm_list.append(lp_hn)

        #FoMoH-KD
        x = sample_init.clone()
        loss_plane = forward_plane_search(x, D = args.plane_dim, fun = rosenbrock_ND, T=T, alpha = args.alpha, softabs_flag=args.softabs, clip_value=args.clip_value)
        lp_hess_plane.append(loss_plane)

        #FGD-lr
        hess_norm = False
        sigma = lr
        samples, lp, iteration, accepts = fomoh_sampler(log_prob_fun, T, sample_init, sigma, hess_norm = hess_norm)
        lp_grad_list_lr.append(lp)

        #FGD-2lr
        sigma = lr * 2
        samples, lp, iteration, accepts = fomoh_sampler(log_prob_fun, T, sample_init, sigma, hess_norm = hess_norm)
        lp_grad_list_2lr.append(lp)

        #ADAM
        loss_list = torch_optim(sample_init, args.lr_adam, T, optim = torch.optim.Adam)
        lp_adam.append(loss_list)

        #Newton
        lp_newton.append(newton_opt(sample_init, rosenbrock_ND, T, alpha = args.alpha, softabs_flag=args.softabs, clip_value=args.clip_value))


        save_dict = {"FoMoH": lp_hess_norm_list, "FoMoH-KD": lp_hess_plane, "FGD-lr": lp_grad_list_lr, "FGD-2lr": lp_grad_list_2lr, "ADAM":lp_adam, "Newton":lp_newton, "Args": args}
    
        torch.save(save_dict, args.save_file)


if __name__ == "__main__":
    main()
    
    
    # python Rosenbrock_ND.py --lr 0.0001 --lr-FoMoH 0.5 --lr-adam 0.2 --epochs 2000 --dim-obj 10 --plane-dim 2 --alpha 0.1 --save-file ./plots/rosenbrock_comparison_obj_dim_10_plane_dim_2.pt