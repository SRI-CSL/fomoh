import torch
from fomoh.hyperdual import HyperTensor as htorch
from fomoh.util import jac_vector_build
from tqdm.notebook import tqdm

def flatten_loss_model(model, loss_module, inputs, labels):
    inputs = htorch(inputs)
    labels = htorch(labels)
    def fun(theta):
        # theta is htorch
        model.sample_to_model(theta.real.flatten())
        if theta.eps1 is None:
            pred = model(inputs, None)
            out = loss_module(pred, labels)
            return pred, out
        else:
            pred = model(inputs, model.vec_to_params(theta.eps1.flatten()), model.vec_to_params(theta.eps2.flatten()))
            out = loss_module(pred, labels)
            return pred, out
    return fun

def flatten_loss_model_batched(model, loss_module, inputs, labels):
    inputs = htorch(inputs)
    labels = htorch(labels)
    def fun(theta):
        # theta is htorch
        model.sample_to_model(theta[0].real.flatten()) ### theta is B x D
        if theta.eps1 is None:
            pred = model(inputs, None)
            out = loss_module(pred, labels)
            return pred, out
        else:
            eps1_list = []
            eps2_list = []
            eps1eps2_list = []
            for v1, v2 in zip(theta.eps1, theta.eps2):
                pred = model(inputs, model.vec_to_params(v1), model.vec_to_params(v2))
                out = loss_module(pred, labels)
                eps1_list.append(out.eps1)
                eps2_list.append(out.eps2)
                eps1eps2_list.append(out.eps1eps2)
                
            out = htorch(out.real.repeat(len(eps1_list)), torch.stack(eps1_list), torch.stack(eps2_list), torch.stack(eps1eps2_list))
            return pred.real, out
    return fun

def flatten_loss_model_batched_vectorized(model, loss_module, inputs, labels):
    inputs = htorch(inputs)
    labels = htorch(labels)
    def fun(theta):
        # theta is htorch
        model.sample_to_model(theta[0].real.flatten()) 
        if theta.eps1 is None:
            pred = model(inputs, None)
            out = loss_module(pred, labels)
            return pred, out
        else:
            V1 = model.batch_vec_to_params(theta.eps1)
            V2 = model.batch_vec_to_params(theta.eps2)
            pred = model(inputs[None], V1, V2) # 1 x 64 x 784 
            out = loss_module(pred, labels[None])
            return pred[-1].real, out
    return fun

def hvp(x, v, fun):
    Hv = torch.zeros(1, x.shape[1])
    J1 = torch.zeros(1, x.shape[1])
    for tensors in jac_vector_build(x.shape[1]):
        xeps1 = tensors.view(1,-1)
        xeps2 = v.clone()
        x_htorch = htorch(x, xeps1, xeps2)
        z = fun(x_htorch)
        Hv[xeps1.bool()] += z.eps1eps2.sum()
        J1[xeps1.bool()] += z.eps1.sum()
    return Hv, J1

def conjugate_gradient(x, v, fun, max_it = 100):
    # https://d2jud02ci9yv69.cloudfront.net/2024-05-07-bench-hvp-81/blog/bench-hvp/
    Hv, b = hvp(x, v.view(1,-1), fun) # Get Jacobian for free (b)
    r = (b - Hv)
    p = r.clone()
    t = 0
    while torch.norm(r) > 1e-3 and t < max_it:
        Hp, bt = hvp(x, p, fun)
        alpha = ((r @ r.t()) / (p @ Hp.t())).flatten().item()
        v += alpha * p
        rt = r - alpha * Hp
        beta = ((rt @ rt.t()) / (r @ r.t())).flatten().item()
        p = rt + beta * p
        r = rt
        t += 1
    return v, t

def plane_step_2d(fun, x, return_real = False, return_pred = False):
    v1 = torch.randn_like(x)
    v2 = torch.randn_like(x)
    V1 = torch.stack([v1, v1, v2])
    V2 = torch.stack([v1, v2, v2])
    x_htorch = htorch(x.repeat(3,1), V1, V2)
    if return_pred:
        pred, z = fun(x_htorch)
    else:
        z = fun(x_htorch)
    H_tilde = torch.tensor([[z.eps1eps2[0], z.eps1eps2[1]],[z.eps1eps2[1], z.eps1eps2[2]]]).to(z.device)
    F_tilde = torch.tensor([z.eps1[0], z.eps1[2]]).view(-1,1).to(z.device)
    H_tilde_inv = torch.linalg.inv(H_tilde)
    kappa = - H_tilde_inv @ F_tilde
    if return_real and return_pred:
        return kappa[0] * v1 + kappa[1] * v2, z.real[0], pred
    elif return_real:
        return kappa[0] * v1 + kappa[1] * v2, z.real[0]
    elif return_pred:
        return kappa[0] * v1 + kappa[1] * v2, pred
    else:
        return kappa[0] * v1 + kappa[1] * v2

    
def invert_matrix(matrix, jitter=1e-5, max_attempts=5, jitter_multiplier=10):
    """
    Attempts to invert a matrix, adding jitter to the diagonal if inversion fails,
    and continues increasing the jitter until the inversion is successful or a maximum number of attempts is reached.
    
    Args:
        matrix (torch.Tensor): The matrix to be inverted.
        jitter (float): The starting amount of jitter to add to the diagonal elements if inversion fails.
        max_attempts (int): The maximum number of attempts to invert the matrix.
        jitter_multiplier (float): The factor by which to increase the jitter after each failed attempt.

    Returns:
        torch.Tensor: The inverted matrix.
    """
    # jitter = initial_jitter
    for attempt in range(max_attempts):
        try:
            # Attempt to invert the matrix
            inv_matrix = torch.linalg.inv(matrix)
            # print("Inversion successful without adding jitter.")
            return inv_matrix
        except RuntimeError as e:
            print(f"Attempt {attempt + 1}: Error detected -", str(e))
            if "singular" in str(e).lower():
                # If inversion fails due to singularity, add jitter to the diagonal
                print(f"Adding jitter to the diagonal: {jitter}")
                jittered_matrix = matrix + jitter * torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
                try:
                    # Attempt to invert the jittered matrix
                    inv_matrix = torch.linalg.inv(jittered_matrix)
                    print(f"Inversion successful after adding jitter: {jitter}")
                    return inv_matrix
                except RuntimeError:
                    # Increase the jitter and try again
                    jitter *= jitter_multiplier
            else:
                # If the error is not due to singularity, re-raise the exception
                raise
    print("Maximum attempts reached. Returning the identity matrix.")
    return torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    

def softabs(H, softabs_const = 1e6):
    eigenvalues, eigenvectors = torch.linalg.eigh(H, UPLO='L')
    abs_eigenvalues = (1./torch.tanh(softabs_const * eigenvalues)) * eigenvalues
    H_s = torch.matmul(eigenvectors, torch.matmul(abs_eigenvalues.diag(), eigenvectors.t()))
    return H_s

def plane_step_Nd(fun, x, N, alpha = 0.0, return_real = False, return_pred = False, softabs_flag = False):
    
    V1 = []
    V2 = []
    
    D = int((N**2 + N)/2)
    
    x_repeat = x.expand(D,-1) #x.repeat(D,1)
    tangents = torch.randn(N, x_repeat.shape[1]).to(x.device)
    
    for i in range(N):
        for j in range(i, N):
            V1.append(tangents[i])
            V2.append(tangents[j])

    x_htorch = htorch(x_repeat, torch.stack(V1), torch.stack(V2))
        
    if return_pred:
        pred, z = fun(x_htorch)
    else:
        z = fun(x_htorch)
    
    
    H_tilde = torch.zeros(N,N).to(z.device)
    F_tilde = torch.zeros(N).view(-1,1).to(z.device)
    k = 0
    for i in range(N):
        F_tilde[i] = z.eps1[k]
        for j in range(i, N):
            H_tilde[i,j] = z.eps1eps2[k]
            H_tilde[j,i] = z.eps1eps2[k] # if i==j then it just overwrites twice   
            k += 1

    if softabs_flag:
        H_tilde = softabs(H_tilde + alpha * torch.eye(N))
    else:
        H_tilde = H_tilde
    H_tilde_inv = invert_matrix(H_tilde, jitter=1e-3)
    
    
    kappa = - H_tilde_inv @ F_tilde
    
    direction = torch.stack([k*v for k,v in zip(kappa, tangents)]).sum(0)
    if return_real and return_pred:
        return direction, z.real[0], pred
    elif return_real:
        return direction, z.real[0]
    elif return_pred:
        return direction, pred
    else:
        return direction
    
    
def optimizer_step_plane_Nd(model, loss_module, n_sample_directions, inputs, labels, N = 2, device = "cpu", clip_value = 0.0, lr = 1.0, vectorized = False):
    loss = 0.
    directional_derivative = torch.zeros(model.n_params).to(device)
    if vectorized:
        fun = flatten_loss_model_batched_vectorized(model, loss_module, inputs, labels)
    else:
        fun = flatten_loss_model_batched(model, loss_module, inputs, labels)
    theta = model.params_to_vec()
    
    for n in range(n_sample_directions):
        direction, loss, pred = plane_step_Nd(fun, theta, N, return_real = True, return_pred = True)
        directional_derivative += direction
        
    directional_derivative /= n_sample_directions
    
    if clip_value != 0.0:
        directional_derivative = torch.clamp(directional_derivative, -abs(clip_value), abs(clip_value))
    
    theta = theta + lr * directional_derivative
    model.sample_to_model(theta)
    return loss, pred.real