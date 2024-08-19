import torch

def compute_dZ_dT(X, Y, dX_dT, dY_dT=None):

    if X.size(-1) != Y.size(-2):
        raise ValueError("Inner dimensions of X and Y must match for multiplication") 

    # Determine output shape
    out_shape = calculate_output_shape(X, Y)
    
    # Initialize output tensor based on dimensions of dX_dT and dY_dT
    if dX_dT is not None:
        dtype = dX_dT.dtype
        device = dX_dT.device
    elif dY_dT is not None:
        dtype = dY_dT.dtype
        device = dY_dT.device
    else:
        dtype = X.dtype  # Default to dtype of X if no gradients are provided
        device = X.device
    
    dZ_dT = torch.zeros(out_shape, dtype=dtype, device=device)
    # Compute the derivative contribution
    if dX_dT is not None:
        dZ_dT += torch.matmul(dX_dT, Y)
    if dY_dT is not None:
        dZ_dT += torch.matmul(X, dY_dT)
                
    return dZ_dT

def calculate_output_shape(X, Y):
    if X.dim() < 2 or Y.dim() < 2:
        raise ValueError("Both X and Y must have at least two dimensions.")
    
    # Check the compatibility of the inner matrix multiplication dimensions
    if X.size(-1) != Y.size(-2):
        raise ValueError("The inner dimensions of X and Y must match for multiplication.")
    
    # Getting the penultimate dimension of X and the last dimension of Y
    m = X.size(-2)
    p = Y.size(-1)
    
    # Determine the maximum batch dimensions (excluding the last two dimensions of each tensor)
    max_batch_dims = []
    X_batch_dims = list(X.shape[:-2])
    Y_batch_dims = list(Y.shape[:-2])
    
    # Padding shorter list with ones for broadcasting
    if len(X_batch_dims) < len(Y_batch_dims):
        X_batch_dims = [1] * (len(Y_batch_dims) - len(X_batch_dims)) + X_batch_dims
    else:
        Y_batch_dims = [1] * (len(X_batch_dims) - len(Y_batch_dims)) + Y_batch_dims
    
    # Taking the maximum across each dimension
    for x_dim, y_dim in zip(X_batch_dims, Y_batch_dims):
        max_batch_dims.append(max(x_dim, y_dim))
    
    # Constructing the final shape
    final_shape = max_batch_dims + [m, p]
    return final_shape

def compute_d2Z_dT1T2(X, Y, X_eps1=None, X_eps2=None, Y_eps1=None, Y_eps2=None, d2X_dTT=None, d2Y_dTT=None):
    """
    Compute the full Hessian of Z(T) = X(T)Y(T) with respect to T, incorporating the second-order derivatives,
    and allowing for cases where derivatives with respect to Y are None and handling different dimensions for T.
    """
    
    if X.size(-1) != Y.size(-2):
        raise ValueError("Inner dimensions of X and Y must match for multiplication") 

    # Determine output shape
    out_shape = calculate_output_shape(X, Y)
    
    # Prepare output tensor
    d2Z_dTT = torch.zeros(out_shape, device=X.device)
    
    # Perform matrix multiplications for the second derivatives
    if d2X_dTT is not None:
        d2Z_dTT += torch.matmul(d2X_dTT, Y)
    if d2Y_dTT is not None:
        d2Z_dTT += torch.matmul(X, d2Y_dTT)
    if X_eps1 is not None and Y_eps1 is not None and X_eps2 is not None and Y_eps2 is not None:
        d2Z_dTT += torch.matmul(X_eps1, Y_eps2) + torch.matmul(X_eps2, Y_eps1)
    
    return d2Z_dTT

def iterate_and_modify_tensors(tensor_list):
    total_elements = sum(t.numel() for t in tensor_list)
    current_global_index = 0
    
    for _ in range(total_elements):
        # Clone and reset all tensors to zero
        modified_tensors = [t.clone().zero_() for t in tensor_list]
        
        temp_index = current_global_index
        for t in modified_tensors:
            num_elements = t.numel()
            if temp_index < num_elements:
                # Manually calculate the indices for multi-dimensional tensors
                indices = []
                for dim_size in reversed(t.shape):
                    indices.append(temp_index % dim_size)
                    temp_index //= dim_size
                indices.reverse()
                
                # Set the selected element to one
                t[tuple(indices)] = 1
                break
            else:
                temp_index -= num_elements
        
        yield modified_tensors
        current_global_index += 1


def P_proj(v1, h):
    P_tilde = v1.view(-1,1) @ v1.view(1,-1) * h
    return P_tilde

def P_proj_control(v1, h, c = -1.):
    # A good value for c seems to be -D
    D = v1.shape[0]
    C = torch.eye(D) * (D + 2)
    x = v1.view(-1,1)
    P_tilde = x @ x.t() * h + c * (x @ x.t() * (x.t() @ x).item() - C)
    return P_tilde

def P_proj_control_diag(v1, h, c = -1.):
    # A good value for c seems to be -D
    D = v1.shape[0]
    C = (torch.ones(D) * (D + 2))
    x = v1
    P_tilde_diag = x ** 2 * h + c * (x ** 2 * (x.t() @ x).item() - C)
    return P_tilde_diag

def projection_to_Hessian(P_tilde):
    D = P_tilde.shape[0]
    A = torch.diag(torch.ones(D)*2) + torch.ones(D,D)
    diag = torch.linalg.solve(A, P_tilde.diag())
    return (1 - torch.eye(D))/2. * P_tilde + diag.diag()

def projection_to_Hessian_diag(P_tilde_diag):
    D = P_tilde_diag.shape[0]
    # A = torch.diag(torch.ones(D)*2) + torch.ones(D,D) 
    
    inverse_row_sum = 1. / (D+2.) # A[0].sum()
    diag_proportion = D * 0.5 + 0.5 # A[0,0] / A[0].sum()
    
    inverse_diag_val = diag_proportion * inverse_row_sum
    inverse_row_off_diag_val = (inverse_row_sum - inverse_diag_val) / (D-1)
    x = torch.zeros(D)
    x[1:] = inverse_row_off_diag_val
    x[0] = inverse_diag_val
    off_P = P_tilde_diag * inverse_row_off_diag_val
    diag = P_tilde_diag * inverse_diag_val
    # diag = torch.zeros(D)
    # for i in range(D):
    #     diag[i] = (torch.roll(x,i) * P_tilde_diag).sum()
    # return diag
    for i in range(D):
        diag[i] = diag[i] + off_P[:i].sum() + off_P[i+1:].sum()
    return diag


def hess_vector_build(N):
    '''
    Generator for use in building full Hessian matrix. It iterates through all the combination of basis matrices
    needed to build a full Hessian. E.g. if N = 2, then the output is:
    
    (tensor([1., 0.]), tensor([1., 0.])) -> H[0,0]
    (tensor([1., 0.]), tensor([0., 1.])) -> H[0,1]
    (tensor([0., 1.]), tensor([1., 0.])) -> H[1,0]
    (tensor([0., 1.]), tensor([0., 1.])) -> H[1,1]
    '''
    for i in range(N):
        for j in range(N):
            x = torch.zeros(N)
            y = torch.zeros(N)

            x[i] = 1.
            y[j] = 1.
            yield x, y

def jac_vector_build(N):
    '''
    Generator for use in building full Jacobian matrix. It iterates through all the basis matrices
    needed to build a full Jacobian. E.g. if N = 2, then the output is:
    
    tensor([1., 0.]) -> J[0]
    tensor([0., 1.]) -> J[1]
    '''
    for i in range(N):
        x = torch.zeros(N)
        x[i] = 1.
        yield x