import torch
import math
import numpy as np
from fomoh.hyperdual import HyperTensor as htorch

class Layer:
    def __init__(self):
        super(Layer, self).__init__()
        self.training = True
        self.params = []
        self.names = []
        
    def __setattr__(self, name, value):
        if isinstance(value, Layer):
            for n, v in zip(value.names, value.params):
                self.params.append(v)
                self.names.append(n)
        
        object.__setattr__(self, name, value)
        
    def __call__(self, x, i):
        pass
    
class Linear(Layer):
    def __init__(self, input_dim, output_dim, bias=True, names = ["W", "b"]):
        super(Linear, self).__init__()
        self.bias = bias
        W = init_kaiming(input_dim, output_dim)
        if self.bias:
            b = init_bias(output_dim)
            self.params = [W, b]
        else:
            self.params = [W]
        self.names = names
        assert(len(self.names) == len(self.params))
    
    def __call__(self, x, p, i):
        if len(x.shape) == 4 and len(p[i].shape) == 3: # Batching for transformer
            assert(x.shape[0] == p[i].shape[0])
            x = x.matmul(p[i].unsqueeze(1))
        else:
            x = x.matmul(p[i])
        i += 1
        if self.bias:
            if len(p[i].shape) == 2:
                B = p[i].shape[0]
                ones = [1]*(len(x.shape)-2)
                p[i] = p[i].view(B,*ones,-1) # For batched addition need to add a dim
            x += p[i]
            i += 1
        return x, i
    
class Conv2d(Layer):
    def __init__(self, out_channels, in_channels, kernel, bias = True, names = ["CW", "Cb"]):
        super(Conv2d, self).__init__()
        self.bias = bias
        self.out_channels = out_channels
        W = weight_init_conv2d(out_channels, in_channels, kernel, kernel)
        if self.bias:
            b = bias_init_conv2d(out_channels, in_channels, kernel, kernel)
            self.params = [W, b]
        else:
            self.params = [W]
        self.names = names
        assert(len(self.names) == len(self.params))
    
    def __call__(self, x, params, i, B=None):
        if len(params[i].shape) == 5: # batch model
            if i == 0 and params[i].shape[0] != x.shape[0]:
                B = params[i].shape[0]
                x = x.repeat(B,1,1,1,1) # Ensure we batch correctly, this needs to be improved
            x = batch_conv2d(x, params[i])
        else:
            x = x.conv2d(params[i])
        i+=1
        if self.bias:
            if len(params[i].shape) == 2: # batch model
                x += params[i].view(B, 1, self.out_channels, 1, 1)
            else:
                x += params[i].view(1, self.out_channels, 1, 1)
            i+=1
        
        return x, i, B

class LayerNorm(Layer):
    def __init__(self, shape, eps=1e-05, names = ["W_ln", "b_ln"]):
        super(LayerNorm, self).__init__()
        self.eps = eps
        dims = - (torch.arange(len(shape)) + 1) # shape is a tuple
        self.dims = tuple(d.item() for d in dims)
        weight = torch.ones(shape)
        bias = torch.zeros(shape) 
        self.params = [weight, bias]
        self.names = names
    
    def __call__(self, x, params, i):
        x = layer_norm(x, params[i], params[i+1], dims = self.dims, eps=self.eps)
        i += 2
        return x, i
    
class Dropout(Layer):
    def __init__(self, p=0.5):
        super(Dropout, self).__init__()
        self.p = p
        
    def dropout_train(self, a):
        mask = torch.bernoulli(torch.full(a.shape, 1-self.p)).to(a.device)
        return a * mask / (1-self.p)

    def dropout_test(self, a):
        return a #* (1-prob)
    
    def __call__(self, x):
        if self.training:
            x = self.dropout_train(x)
        else:
            x = self.dropout_test(x)
        return x
    
def init_uniform(shape, k):
    return -k + torch.rand(shape) * 2*k

def init_bias(out_features):
    k = 1 / math.sqrt(out_features)
    return init_uniform(out_features, k)

def init_kaiming(in_features, out_features):
    a = math.sqrt(5.)
    w = torch.randn(in_features, out_features)
    s = math.sqrt(2. / ((1. + a*a) * in_features))
    return w * s

def weight_init_conv2d(out_channels, in_channels, kernel_height, kernel_width):
    w = torch.zeros(out_channels, in_channels, kernel_height, kernel_width)
    torch.nn.init.kaiming_uniform_(w, a=math.sqrt(5))
    return w

def bias_init_conv2d(out_channels, in_channels, kernel_height, kernel_width):
    num_input_fmaps = in_channels
    receptive_field_size = kernel_height*kernel_width
    fan_in = num_input_fmaps*receptive_field_size
    bound = 1 / math.sqrt(fan_in)
    b = torch.zeros(out_channels)
    torch.nn.init.uniform_(b, -bound, bound)
    return b
    
def batch_conv2d(input, weight, *args, **kwargs):
    if len(weight.shape) != 5:
        raise ValueError('Expecting a 5d tensor with shape BxOutCxInCxKxK')
    B, out_c, in_c, K1, K2 = weight.shape
    
    in_shape = input.shape
    if len(in_shape) != 5:
        raise ValueError('Expecting a 5d tensor with shape Bxdata_batchxChannelsxDxD')
    if in_shape[0] != B:
        raise ValueError(f'Expecting outer batch of params: {B}, to equal input outer batch: {input.shape[0]}')
    input = input.movedim(0,1).reshape(in_shape[1], in_shape[2] * B, *in_shape[3:])
    output = input.conv2d(weight.reshape(B*out_c, in_c, K1, K2), *args, groups=B, **kwargs)
    return output.view(in_shape[1],B,out_c,output.shape[-2],output.shape[-1]).movedim(1,0)

def variance(a, dim, unbiased, keepdim=False):
    if isinstance(dim, tuple):
        m = a.mean(dim=dim, keepdim=True)
        a2 = a - m
        if unbiased:
            n = np.prod([a.shape[d] for d in dim]) - 1
        else:
            n = np.prod([a.shape[d] for d in dim])
        a2 = a2 ** 2
        for d in sorted(dim, reverse=True):
            a2 = a2.sum(d, keepdim=True)
        if not keepdim:
            a2 = a2.squeeze()
        return a2 / float(n)
    else:
        if unbiased:
            n = a.shape[dim] - 1
        else:
            n = a.shape[dim]
        a2 = a - a.mean(dim=dim, keepdim=True)
        return (a2 ** 2).sum(dim=dim, keepdim=True) / n

def layer_norm(x, weight, bias, dims = -1, eps=1e-05):
    N = len(x.shape)
    if isinstance(dims, tuple):
        D = len(dims)
        shapes = [x.shape[d] for d in np.sort(dims)]
    else:
        shapes = [x.shape[dims]]
        D = 1
    if len(weight.shape) > D: # batch
        B = weight.shape[0]
        ones = [B] + [1] * (N - D - 1)
    else:
        ones = [1] * (N - D)
    m, v = x.mean(dims, keepdim=True), variance(x, dim=dims, unbiased=False, keepdim=True)
    norm = ((x - m)/((v + eps)**0.5)) * weight.view(*ones,*shapes) + bias.view(*ones,*shapes)
    return norm

def batch_layer_norm(x, weight, bias, dims = -1, eps=1e-05):
    N = len(x.shape)
    B = x.shape[0]
    if isinstance(dims, tuple):
        D = len(dims)
        shapes = [x.shape[d] for d in np.sort(dims)]
    else:
        shapes = [x.shape[dims]]
        D = 1
    ones = [1] * (N - D - 1)
    m, v = x.mean(dims, keepdim=True), variance(x, dim=dims, unbiased=False, keepdim=True)
    norm = ((x - m)/((v + eps)**0.5)) * weight.view(B, *ones,*shapes) + bias.view(B, *ones,*shapes)
    return norm

def dropout_train(a, prob=0.5):
    mask = torch.bernoulli(torch.full(a.shape, 1-prob)).to(a.device)
    return a * mask / (1-prob)

def dropout_test(a, prob=0.5):
    return a 
