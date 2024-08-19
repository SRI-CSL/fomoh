import torch
from fomoh.hyperdual import HyperTensor as htorch
import torch.nn as nn
from fomoh.util import P_proj_control, projection_to_Hessian, P_proj_control_diag, projection_to_Hessian_diag
from fomoh.layers import Layer, Linear, Conv2d, init_bias, init_kaiming, weight_init_conv2d, bias_init_conv2d, batch_conv2d, LayerNorm, Dropout
from collections import OrderedDict

def update_or_add_key(ordered_dict, key, new_value):
    """
    Updates the value of a key in an OrderedDict if the key exists.
    Otherwise, it adds the key with the new value to the end of the OrderedDict.

    Parameters:
    - ordered_dict: The OrderedDict to update.
    - key: The key to update or add.
    - new_value: The new value for the key.
    """
    if key in ordered_dict:
        # If the key exists, update its value
        ordered_dict[key] = new_value
    else:
        # If the key does not exist, add it to the end
        ordered_dict.update({key: new_value})

class Model:
    def __init__(self):
        super(Model, self).__init__()
        self.params = OrderedDict([])
        self.named_params = []
        self.device = "cpu"
        self.training = True
        self.n_params = self.count_n_params()
        
    def __setattr__(self, name, value):
        if isinstance(value, Layer):
            for n, v in zip(value.names, value.params):
                self.add_param(n, v)
        
        object.__setattr__(self, name, value)
        
    def __repr__(self):
        parts = [f"{key}: {value.shape} \n" for key, value in self.params.items()]
        return "Model(\n" + "".join(parts) + ")"
    
    def add_param(self, name, params):
        setattr(self, name, torch.nn.Parameter(params, requires_grad=False))
        update_or_add_key(self.params, name, getattr(self, name))
        self.n_params = self.count_n_params()
        self.named_params = list(self.params.keys())
    
    @staticmethod
    def set_training(layer, flag):
        layer.training = flag
        # Recursively set training for all nested Layer instances
        for attr_name in layer.__dict__:
            attr_value = getattr(layer, attr_name)
            if isinstance(attr_value, Layer):
                Model.set_training(attr_value, flag)
    
    def eval(self):
        Model.set_training(self, False)

    def train(self):
        Model.set_training(self, True)
                
    def count_n_params(self):
        n = 0
        for p in self.params.values():
            n += torch.tensor(p.shape).prod()
        return n
    
    def vec_to_params(self, vec):
        vec_reshape = []
        i = 0
        for p in self.params.values():
            n = torch.tensor(p.shape).prod()
            vec_reshape.append(vec[i:i+n].view(p.shape))
            i += n
        return vec_reshape

    def batch_vec_to_params(self, vec):
        vec_reshape = []
        i = 0
        for p in self.params.values():
            n = torch.tensor(p.shape).prod()
            vec_reshape.append(vec[:,i:i+n].view(-1, *p.shape))
            i += n
        return vec_reshape

    def params_to_vec(self, v = None):
        vec = []
        if v is None:
            for p in self.params.values():
                vec.append(p.reshape(-1))
        else:
            for p in v:
                vec.append(p.reshape(-1))
        return torch.cat(vec)

    def convert_params_to_htorch(self, v1, v2, requires_grad = False):
        hparams = []
        # Check for tangent batch:
        if v1[0] is None:
            batch = 0
        else:
            if len(v1[0].shape) > len(list(self.params.values())[0].shape):
                batch = v1[0].shape[0]
            else:
                batch = 0
        for p, eps1, eps2 in zip(self.params.values(), v1, v2):
            if batch == 0:
                eps1eps2 = torch.zeros_like(eps1) if eps1 is not None else None
                hparams.append(htorch(p.requires_grad_(requires_grad), eps1, eps2, eps1eps2))
            else:
                eps1eps2 = torch.zeros_like(eps1) if eps1 is not None else None
                hparams.append(htorch(p[None].repeat(batch, *[1 for _ in range(len(p.shape))]).requires_grad_(requires_grad), eps1, eps2, eps1eps2))
        return hparams
    
    def nn_module_to_htorch_model(self, model, verbose = True):
        for (module_name, module_params), htorch_name  in zip(model.named_parameters(), self.named_params):
            if len(self.params[htorch_name].shape) == 2: # Linear weight
                if module_params.data.t().shape==self.params[htorch_name].shape:
                    self.add_param(htorch_name, module_params.data.t())
                else:
                    #Embedding
                    self.add_param(htorch_name, module_params.data)
            elif len(self.params[htorch_name].shape) == 4: # Conv2D layer
                assert(module_params.data.shape==self.params[htorch_name].shape)
                self.add_param(htorch_name, module_params.data)
            elif len(self.params[htorch_name].shape) == 1: # bias weight
                assert(module_params.data.shape==self.params[htorch_name].shape)
                self.add_param(htorch_name, module_params.data)
            else:
                raise NotImplementedError("Shape of weight means not implemented layer: " + htorch_name)
        if verbose:
            print("Weights transferred to htorch model.")
    
    def collect_nn_module_grads(self, model):
        gradients = []
        for (module_name, module_params), htorch_name  in zip(model.named_parameters(), self.named_params):
            if len(self.params[htorch_name].shape) == 2: # Linear weight
                if module_params.grad.t().shape==self.params[htorch_name].shape:
                    gradients.append(module_params.grad.t())
                else:
                    #Embedding
                    gradients.append(module_params.grad)
            elif len(self.params[htorch_name].shape) == 4: # Conv2D layer
                assert(module_params.grad.shape==self.params[htorch_name].shape)
                gradients.append(module_params.grad)
            elif len(self.params[htorch_name].shape) == 1: # bias weight
                assert(module_params.grad.shape==self.params[htorch_name].shape)
                gradients.append(module_params.grad)
            else:
                raise NotImplementedError("Shape of weight means not implemented layer: " + htorch_name)
        return gradients
        
    def sample_to_model(self, sample):
        '''
        Sample is a flat vector that needs converting
        '''
        new_params = self.vec_to_params(sample)
        for name, p in zip(self.named_params, new_params):
            self.add_param(name, p)

    def to(self, device):
        self.device = device
        for key in self.params:
            self.params[key] = self.params[key].to(device)
    
    def type(self, dtype):
        for key in self.params:
            self.params[key] = self.params[key].type(dtype=dtype)
            
    
class LogisticRegressionModel(Model):
    def __init__(self, input_dim, output_dim, bias = True):
        super(LogisticRegressionModel, self).__init__()
        self.linear = Linear(input_dim, output_dim, bias)
        self.n_params = self.count_n_params() # Not needed anymore
        
    def __call__(self, x, v1, v2=None, requires_grad=False):
        if v1 is None:
            v1 = [None for _ in self.params]
            v2 = [None for _ in self.params]
        elif v2 is None:
            v2 = v1 #[v.clone() for v in v1]
        params = self.convert_params_to_htorch(v1,v2, requires_grad=requires_grad)
        
        x, i = self.linear(x, params, 0)
        
        return x.logsoftmax(-1)
    
class DenseModel(Model):
    def __init__(self, layers = [1,100,1], bias = True):
        super(DenseModel, self).__init__()
        self.layers = layers
        self.bias = bias
        self.linear_layers = []
        for n in range(len(self.layers)-1):
            if self.bias:
                self.__setattr__(f'linear_{n+1}', Linear(self.layers[n], self.layers[n+1], bias, names = [f'W{n+1}', f'b{n+1}']))
            else:
                self.__setattr__(f'linear_{n+1}', Linear(self.layers[n], self.layers[n+1], bias, names = [f'W{n+1}']))
            self.linear_layers.append(getattr(self, f'linear_{n+1}'))
                
        self.n_params = self.count_n_params()
        
    def __call__(self, x, v1, v2=None, requires_grad=False):
        if v1 is None:
            v1 = [None for _ in self.params]
            v2 = [None for _ in self.params]
        elif v2 is None:
            v2 = v1 #[v.clone() for v in v1]
        params = self.convert_params_to_htorch(v1, v2, requires_grad=requires_grad)
        
        i = 0
        for n in range(len(self.layers)-1):
            x, i = self.linear_layers[n](x, params, i)
            if n < ( len(self.layers) - 2):
                # x = x.sigmoid()
                x = x.relu()
        return x#.logsoftmax()

class CNNModel(Model):
    def __init__(self, cnn_layers_channels = [1,20,50], cnn_filter_size = 5, dense_layers = [4*4*50,500,10], maxpool_args = [2,2], bias = True):
        super(CNNModel, self).__init__()
        self.cnn_channels = cnn_layers_channels
        self.filter = cnn_filter_size
        self.dense_layers = dense_layers
        self.bias = bias
        self.maxpool_args = maxpool_args
        self.conv2d_layers = []
        for n in range(len(self.cnn_channels)-1):
            if self.bias:
                self.__setattr__(f'cnn_{n+1}', Conv2d(self.cnn_channels[n+1], self.cnn_channels[n], self.filter, bias, names = [f'CW{n+1}', f'Cb{n+1}']))
            else:
                self.__setattr__(f'cnn_{n+1}', Conv2d(self.cnn_channels[n+1], self.cnn_channels[n], self.filter, bias, names = [f'CW{n+1}']))
            self.conv2d_layers.append(getattr(self, f'cnn_{n+1}'))
        
        self.linear_layers = []
        for n in range(len(self.dense_layers)-1):
            if self.bias:
                self.__setattr__(f'linear_{n+1}', Linear(self.dense_layers[n], self.dense_layers[n+1], bias, names = [f'W{n+1}', f'b{n+1}']))
            else:
                self.__setattr__(f'linear_{n+1}', Linear(self.dense_layers[n], self.dense_layers[n+1], bias, names = [f'W{n+1}']))
            self.linear_layers.append(getattr(self, f'linear_{n+1}'))
                
        self.n_params = self.count_n_params()
        
    def __call__(self, x, v1, v2=None, requires_grad=False):
        if v1 is None:
            v1 = [None for _ in self.params]
            v2 = [None for _ in self.params]
        elif v2 is None:
            v2 = v1 #[v.clone() for v in v1]
        params = self.convert_params_to_htorch(v1,v2, requires_grad=requires_grad)
        
        i = 0
        B = None
        for n in range(len(self.cnn_channels)-1):
            x, i, B = self.conv2d_layers[n](x, params, i, B)
            x = x.relu()
            
            if len(x.shape) == 5: # Batching
                x = x.reshape(-1, *x.shape[2:]).maxpool2d(*self.maxpool_args)
                x = x.reshape(B, -1, *x.shape[1:])
            else:
                x = x.maxpool2d(*self.maxpool_args)

        if B is None:
            x = x.view(-1, self.dense_layers[0])
        else:
            x = x.view(B, -1, self.dense_layers[0])
        
        for n in range(len(self.dense_layers)-1):
            x, i = self.linear_layers[n](x, params, i)
            if n < (len(self.dense_layers) - 2):
                # x = x.sigmoid()
                x = x.relu()
                
        return x#.logsoftmax()

def nll_loss_old(input, target, reduce = "mean"):
    if input.real.dim() != 2 or target.real.dim() != 1:
        raise ValueError('Expecting 2d input and 1d target')
    n, c = input.real.shape[0], input.real.shape[1]
    l = 0.
    for i in range(n):
        t = int(target.real[i])
        l -= input[i, t]
    if reduce == "mean":
        return l / n
    else: # sum
        return l

def nll_loss(input, target, reduce="mean", ignore_index = None):
    # Check if input is batched
    if len(input.shape) == 3:  # Assuming batched input is 3D
        batched = True
        batch_size, n, c = input.shape
    elif len(input.shape) == 2:  # Non-batched input is 2D
        batched = False
        n, c = input.shape
        input = input.unsqueeze(0)  # Add a batch dimension of 1 for uniformity
        target = target.unsqueeze(0)
    else:
        raise ValueError('Expecting 2D input for non-batched or 3D input for batched processing')

    # Ensure target is compatible
    if len(target.shape) not in {1, 2} or (batched and len(target.shape) == 1):
        raise ValueError('Target dimension mismatch')

    if len(target.shape) == 1:
        target = target.view(1, -1)  # Ensure target is 2D if it's not

    # Using advanced indexing to select the relevant probabilities
    # Create a range tensor for indexing rows: [0, 1, 2, ..., n-1] (or batched indices)
    if batched:
        batch_indices = torch.arange(batch_size).unsqueeze(1).expand(batch_size, n).to(target.device)
        loss = -input[batch_indices, torch.arange(n).unsqueeze(0).expand(batch_size, n), target.real]
    else:
#         import pdb; pdb.set_trace()
        loss = -input[0, torch.arange(n), target[0].real]  # Only one batch

    if ignore_index is not None:
        # Create a mask for elements not to be ignored
        mask = (target.real != ignore_index).squeeze()#.float()
        loss = loss[mask]  # Apply mask

        # Reduce the loss as specified
        if reduce == "mean":
            return loss.sum(-1) / mask.sum()# if batched else loss.mean(dim=1)
        elif reduce == "sum":
            return loss.sum(-1)# if batched else loss.sum(dim=1)
        else:
            return loss
    else:
        # Reduce the loss as specified
        if reduce == "mean":
            return loss.mean(-1)# if batched else loss.mean(dim=1)
        elif reduce == "sum":
            return loss.sum(-1)# if batched else loss.sum(dim=1)
        else:
            return loss
        # raise ValueError("reduce argument must be either 'mean' or 'sum'")
    
def mean(a, dim, keepdim=False):
    sm = a.sum(dim, keepdim=keepdim)
    dv = sm / a.shape[dim]
    return dv

def variance(a, dim, unbiased, keepdim=False):
    # This is the two-pass algorithm, see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    if unbiased:
        n = a.shape[dim] - 1
    else:
        n = a.shape[dim]
    a2 = a - mean(a, dim=dim, keepdim=True)
    return (a2 ** 2).sum(dim=dim, keepdim=keepdim) / n

def batchnorm2d(a, weight, bias, eps=1e-05):
    if len(a.shape) != 4:
        raise ValueError('Expecting a 4d tensor with shape BxCxHxW')
    num_features = a.shape[1]
    at = a.transpose(0, 1).reshape(num_features, -1)
    m, v = mean(at, dim=1), variance(at, dim=1, unbiased=False)
    res = (a - m.view(1, num_features, 1, 1)) / ((v.view(1, num_features, 1, 1) + eps) ** 0.5 )
    res = res * weight.view(1, num_features, 1, 1) + bias.view(1, num_features, 1, 1)
    return res

def update_model_gradients_(model, external_derivatives):
    # Manually set the .grad attributes of the model's parameters
    for k, external_grad in zip(model.params.keys(), external_derivatives):
        model.params[k].grad = external_grad

def update_model_parameters(params, update_term):
    # Add new gradient term
    for param, p_up in zip(params, update_term):
        param += p_up

def tangent_dropout_mask(x, p=0.1):
    return (torch.rand_like(x) > p)
    

def optimizer_step(model, loss_module, optimizer, n_sample_directions, input, target, device = "cpu", clip_value = 0.0, hess = False, epsilon = 10e-7, tangent_dropout = 0.0, backprop = False):
    loss = 0.
    directional_derivative = torch.zeros(model.n_params).to(device)
    ### Might be able to parallelize this!
    if backprop:
        assert(n_sample_directions==1)
    
    for n in range(n_sample_directions):
        if backprop:
            # Zero gradients:
            for p in model.params.values():
                p.grad = None
            pred = model(input, None, requires_grad=True)
            out = loss_module(pred, target)
            out.real.backward()
            param_directions = []
            for p in model.params.values():
                param_directions.append(p.grad)
            param_directions = model.params_to_vec(param_directions)
            
        else:
            param_directions = torch.randn(model.n_params).to(device)
            if tangent_dropout != 0.0:
                param_directions *= tangent_dropout_mask(param_directions, p=tangent_dropout)
            # param_directions =param_directions/param_directions.norm()
        param_directions_reshaped = model.vec_to_params(param_directions)
        pred = model(input, param_directions_reshaped)
        out = loss_module(pred, target)
        if hess:
            directional_derivative += out.eps1.item() * param_directions / (abs(out.eps1eps2.item()) + epsilon)
        else:
            directional_derivative += out.eps1.item() * param_directions

    loss = out#.real.item() # Loss should be the same at each iteration as only tangents are changing
    directional_derivative /= n_sample_directions
    
    optimizer.zero_grad()
    
    if clip_value != 0.0:
        directional_derivative = torch.clamp(directional_derivative, -abs(clip_value), abs(clip_value))
    grads = model.vec_to_params(directional_derivative)
    
    update_model_gradients_(model, grads)
    optimizer.step()
    return loss, pred.real

def evaluate_model(input, target, param_directions, model, loss_module):
    param_directions_reshaped = model.vec_to_params(param_directions)
    pred = model(input, param_directions_reshaped)
    out = loss_module(pred, target)
    return out, pred


def newton_step(model, loss_module, n_sample_directions, input, target, lr=1.0, control = 0., epsilon = 1e-5, beta = 1.0):
    loss = 0.
    directional_derivative = torch.zeros(model.n_params)
    projection_outer_product = torch.zeros(model.n_params, model.n_params)
    for n in range(n_sample_directions):
        param_directions = torch.randn(model.n_params)
        param_directions_reshaped = model.vec_to_params(param_directions)
        pred = model(input, param_directions_reshaped)
        out = loss_module(pred, target)
        loss += out.real.item()
        directional_derivative += out.eps1.item() * param_directions
        projection_outer_product += P_proj_control(param_directions, out.eps1eps2.item(), c = control)

    loss /= n_sample_directions
    directional_derivative /= n_sample_directions
    projection_outer_product /= n_sample_directions
    # Add some jitter
    projection_outer_product += torch.eye(model.n_params) * epsilon
    # print(projection_outer_product)

    ### Full Newton's step:
    H_tilde = projection_to_Hessian(projection_outer_product)
    t = - lr * (beta * torch.linalg.inv(H_tilde) @ directional_derivative.view(-1,1) + (1.-beta) * directional_derivative.view(-1,1))

    additive_update = model.vec_to_params(t)
    
    update_model_parameters(model.params, additive_update)
    
    return loss

def newton_step_diag(model, loss_module, n_sample_directions, input, target, lr=1.0, control = 0., epsilon = 1e-5, beta = 1.0, device = "cpu"):
    loss = 0.
    directional_derivative = torch.zeros(model.n_params).to(device)
    projection_outer_product = torch.zeros(model.n_params).to(device)
    for n in range(n_sample_directions):
        param_directions = torch.randn(model.n_params).to(device)
        param_directions_reshaped = model.vec_to_params(param_directions)
        pred = model(input, param_directions_reshaped)
        out = loss_module(pred, target)
        loss += out.real.item()
        # print("1")
        directional_derivative += out.eps1.item() * param_directions
        projection_outer_product += P_proj_control_diag(param_directions, out.eps1eps2.item(), c = control)
    # print("2")

    loss /= n_sample_directions
    directional_derivative /= n_sample_directions
    projection_outer_product /= n_sample_directions
    # Add some jitter

    ### Diag Newton's step:
    H_tilde_diag = projection_to_Hessian_diag(projection_outer_product)
    # print("4")
    t = - lr * (beta * directional_derivative.view(-1,1) / H_tilde_diag.view(-1,1)  + (1.-beta) * directional_derivative.view(-1,1))

    additive_update = model.vec_to_params(t)
    
    update_model_parameters(model.params, additive_update)
    
    return loss
