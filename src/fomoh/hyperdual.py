import torch
from fomoh.util import compute_dZ_dT, compute_d2Z_dT1T2

def check_first_order(A):
    if (A.eps1 is None and A.eps2 is not None) or (A.eps1 is not None and A.eps2 is None):
        raise ValueError(f"A.eps1: {A.eps1}, while A.eps2: {A.eps2}")

def make_tensor(obj, device = "cpu"):
    if torch.is_tensor(obj):
        return obj
    elif obj is None:
        return None
    else:
        # Only put on device if it is not already a tensor
        return torch.tensor(obj).to(device)

class HyperTensor:
    def __init__(self, real, eps1=None, eps2=None, eps1eps2=None):
        self.real = make_tensor(real)
        self.device = self.real.device
        self.eps1 = make_tensor(eps1, self.device)
        self.eps2 = make_tensor(eps2, self.device)
        self.eps1eps2 = make_tensor(eps1eps2, self.device)

    def __repr__(self):
        parts = [
            f"real={self.real}",
            f"eps1={self.eps1}" if self.eps1 is not None else "eps1=None",
            f"eps2={self.eps2}" if self.eps2 is not None else "eps2=None",
            f"eps1eps2={self.eps1eps2}" if self.eps1eps2 is not None else "eps1eps2=None"
        ]
        return "HyperTensor(" + ", ".join(parts) + ")"
    
    @property
    def shape(self):
        return self.real.shape
    
    @staticmethod
    def binary(A, B, f, fa, fb, f1all, faa, fbb, f2all):
        real = f(A, B)
        check_first_order(A) #Ensures that if eps1 is not None then eps2 is not None
        check_first_order(B)
        if A.eps1 is None:
            if B.eps1 is None:
                eps1 = None
                eps2 = None
                eps1eps2 = None
            else:
                eps1 = fb(real, A, B, A.eps1, B.eps1)
                eps2 = fb(real, A, B, A.eps2, B.eps2)
                eps1eps2 = fbb(real, A, B)
        elif B.eps1 is None:
            eps1 = fa(real, A, B, A.eps1, B.eps1)
            eps2 = fa(real, A, B, A.eps2, B.eps2)
            eps1eps2 = faa(real, A, B)
        else: # Needs to account for jac existing but no hessian
            eps1 = f1all(real, A, B, A.eps1, B.eps1)
            eps2 = f1all(real, A, B, A.eps2, B.eps2)
            eps1eps2 = f2all(real, A, B)
        return HyperTensor(real, eps1, eps2, eps1eps2)

    @staticmethod
    def unary(A, f, fa, faa):
        real = f(A.real)
        check_first_order(A)
        if A.eps1 is None:
            eps1 = None
            eps2 = None
            eps1eps2 = None
        else:
            eps1 = fa(real, A, A.eps1)
            eps2 = fa(real, A, A.eps2)
            eps1eps2 = faa(real, A)
        if isinstance(real, tuple):
            # For multiple outputs like torch.split
            if eps1 is None:
                eps1 = (None for _ in range(len(real)))
            if eps2 is None:
                eps2 = (None for _ in range(len(real)))
            if eps1eps2 is None:
                eps1eps2 = (None for _ in range(len(real)))
                
            return (HyperTensor(r, e1, e2, e1e2) for r, e1, e2, e1e2 in zip(real, eps1, eps2, eps1eps2))
        else:
            return HyperTensor(real, eps1, eps2, eps1eps2)
        
    def __add__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            # Handle scalar addition
            other = HyperTensor(other)
        f = lambda A, B: A.real + B.real
        fa = lambda real, A, B, A_eps1, B_eps1 : A_eps1
        fb = lambda real, A, B, A_eps1, B_eps1 : B_eps1
        f1all = lambda real, A, B, A_eps1, B_eps1: A_eps1 + B_eps1
        faa = lambda real, A, B: None if A.eps1eps2 is None else A.eps1eps2
        fbb = lambda real, A, B: None if B.eps1eps2 is None else B.eps1eps2
        f2all = lambda real, A, B: None if A.eps1eps2 is None and B.eps1eps2 is None else ((A.eps1eps2 if A.eps1eps2 is not None else 0.) + (B.eps1eps2 if B.eps1eps2 is not None else 0.))
        return HyperTensor.binary(self, other, f, fa, fb, f1all, faa, fbb, f2all)
    
    def __radd__(self, other):
        # Addition is commutative
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            # Handle scalar subtraction
            other = HyperTensor(other)
        f = lambda A, B: A.real - B.real
        fa = lambda real, A, B, A_eps1, B_eps1 : A_eps1
        fb = lambda real, A, B, A_eps1, B_eps1 : - B_eps1
        f1all = lambda real, A, B, A_eps1, B_eps1: A_eps1 - B_eps1
        faa = lambda real, A, B: None if A.eps1eps2 is None else A.eps1eps2
        fbb = lambda real, A, B: None if B.eps1eps2 is None else - B.eps1eps2
        f2all = lambda real, A, B: None if A.eps1eps2 is None and B.eps1eps2 is None else ((A.eps1eps2 if A.eps1eps2 is not None else 0.) - (B.eps1eps2 if B.eps1eps2 is not None else 0.))
        return HyperTensor.binary(self, other, f, fa, fb, f1all, faa, fbb, f2all)

    def __rsub__(self, other):
        # For reverse subtraction, we need to negate self and then add
        if isinstance(other, (int, float, torch.Tensor)):
            other = HyperTensor(other)
        return other - self  

    def __mul__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            # Handle scalar multiplication
            other = HyperTensor(other)
        f = lambda A, B: A.real * B.real
        fa = lambda real, A, B, A_eps1, B_eps1 : A_eps1 * B.real
        fb = lambda real, A, B, A_eps1, B_eps1 : A.real * B_eps1
        f1all = lambda real, A, B, A_eps1, B_eps1 : A_eps1 * B.real + A.real * B_eps1
        faa = lambda real, A, B: None if A.eps1eps2 is None else A.eps1eps2 * B.real
        fbb = lambda real, A, B: None if B.eps1eps2 is None else A.real * B.eps1eps2
        def f2all(real, A, B):
            if A.eps1eps2 is None:
                if B.eps1eps2 is None:
                    return A.eps1 * B.eps2 + A.eps2 * B.eps1
                else:
                    return A.eps1 * B.eps2 + A.eps2 * B.eps1 + A.real * B.eps1eps2
            elif B.eps1eps2 is None:
                return A.eps1eps2 * B.real + A.eps1 * B.eps2 + A.eps2 * B.eps1
            else:
                return A.eps1eps2 * B.real + A.eps1 * B.eps2 + A.eps2 * B.eps1 + A.real * B.eps1eps2
        return HyperTensor.binary(self, other, f, fa, fb, f1all, faa, fbb, f2all)

    def __rmul__(self, other):
        # Multiplication is commutative
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            if other == 0:
                raise ZeroDivisionError("Division by zero in the real component of HyperTensor.")
        if isinstance(other, (int, float, torch.Tensor)):
            # Handle scalar multiplication
            other = HyperTensor(other)
        f = lambda A, B: A.real / B.real
        fa = lambda real, A, B, A_eps1, B_eps1: A_eps1 / B.real
        fb = lambda real, A, B, A_eps1, B_eps1: - (real / B.real) * B_eps1
        f1all = lambda real, A, B, A_eps1, B_eps1: (A_eps1 - B_eps1 * real) / B.real
        faa = lambda real, A, B: None if A.eps1eps2 is None else A.eps1eps2 / B.real
        fbb = lambda real, A, B: (A.real * (B.eps1 * B.eps2 + B.eps2 * B.eps1 - (0. if B.eps1eps2 is None else B.real * B.eps1eps2)))/ B.real**3
        def f2all(real, A, B):
            if A.eps1eps2 is None:
                if B.eps1eps2 is None:
                    return (- B.real * (A.eps1 * B.eps2 + A.eps2 * B.eps1) + A.real *( B.eps1 * B.eps2 + B.eps2 * B.eps1)) / B.real ** 3
                else:
                    return ( - B.real * ((A.eps1 * B.eps2 + A.eps2 * B.eps1) + A.real * B.eps1eps2) + A.real *( B.eps1 * B.eps2 + B.eps2 * B.eps1)) / B.real ** 3
            elif B.eps1eps2 is None:
                return (B.real ** 2 * A.eps1eps2 - B.real * ((A.eps1 * B.eps2 + A.eps2 * B.eps1)) + A.real *( B.eps1 * B.eps2 + B.eps2 * B.eps1)) / B.real ** 3
            else:
                return (B.real ** 2 * A.eps1eps2 - B.real * ((A.eps1 * B.eps2 + A.eps2 * B.eps1) + A.real * B.eps1eps2) + A.real *( B.eps1 * B.eps2 + B.eps2 * B.eps1)) / B.real ** 3
        return HyperTensor.binary(self, other, f, fa, fb, f1all, faa, fbb, f2all)

    def __rtruediv__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            # Handle scalar division
            other = HyperTensor(other)
        return other.__truediv__(self)
    
    def __neg__(self):
        f = lambda A: - A.real
        fa = lambda real, A, A_eps1: - A_eps1
        faa = lambda real, A: - A.eps1eps2 if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def matmul(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            # Handle scalar multiplication
            other = HyperTensor(other)
        f = lambda A, B: torch.matmul(A.real, B.real)
        fa = lambda real, A, B, A_eps1, B_eps1: compute_dZ_dT(A.real, B.real, A_eps1)
        fb = lambda real, A, B, A_eps1, B_eps1: compute_dZ_dT(A.real, B.real, None, B_eps1)
        f1all = lambda real, A, B, A_eps1, B_eps1: compute_dZ_dT(A.real, B.real, A_eps1, B_eps1)
        faa = lambda real, A, B: compute_d2Z_dT1T2(A.real, B.real, A.eps1, A.eps2, None, None, A.eps1eps2, None)
        fbb = lambda real, A, B: compute_d2Z_dT1T2(A.real, B.real, None, None, B.eps1, B.eps2, None, B.eps1eps2)
        f2all = lambda real, A, B: compute_d2Z_dT1T2(A.real, B.real, A.eps1, A.eps2, B.eps1, B.eps2, A.eps1eps2, B.eps1eps2)
        return HyperTensor.binary(self, other, f, fa, fb, f1all, faa, fbb, f2all)

    def sin(self):
        f = lambda A: torch.sin(A.real)
        fa = lambda real, A, A_eps1: torch.cos(A.real) * A_eps1 
        faa = lambda real, A: (torch.cos(A.real) * A.eps1eps2 if A.eps1eps2 is not None else 0.) - torch.sin(A.real) * A.eps1 * A.eps2 
        return HyperTensor.unary(self, f, fa, faa)

    def exp(self):
        f = lambda A: torch.exp(A.real)
        fa = lambda real, A, A_eps1: torch.exp(A.real) * A_eps1
        faa = lambda real, A: torch.exp(A.real) * ((A.eps1eps2 if A.eps1eps2 is not None else 0.) + A.eps1*A.eps2)
        return HyperTensor.unary(self, f, fa, faa)
    
    def tanh(self):
        f = lambda A: torch.tanh(A.real)
        fa = lambda real, A, A_eps1: (1/torch.cosh(A.real)) **2 * A_eps1
        faa = lambda real, A: (1/torch.cosh(A.real)) **2 * ((A.eps1eps2 if A.eps1eps2 is not None else 0.) - 2 * A.eps1*A.eps2 * torch.tanh(A.real))
        return HyperTensor.unary(self, f, fa, faa)

    def log(self):
        f = lambda A: torch.log(A.real)
        fa = lambda real, A, A_eps1: (1./A.real) * A_eps1
        faa = lambda real, A: (1./A.real ** 2) * ((A.real * A.eps1eps2 if A.eps1eps2 is not None else 0.) - A.eps1*A.eps2)
        return HyperTensor.unary(self, f, fa, faa)
    
    def relu(self):
        f = lambda A: A.real.relu()
        fa = lambda real, A, A_eps1: A_eps1 * (A.real > 0)
        faa = lambda real, A: (A.real > 0) * A.eps1eps2 if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def sum(self, *args, **kwargs):
        f = lambda A: A.real.sum(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.sum(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.sum(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def mean(self, *args, **kwargs):
        f = lambda A: A.real.mean(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.mean(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.mean(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def squeeze(self, *args, **kwargs):
        f = lambda A: A.real.squeeze(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.squeeze(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.squeeze(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)
    
    def unsqueeze(self, *args, **kwargs):
        f = lambda A: A.real.unsqueeze(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.unsqueeze(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.unsqueeze(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def view(self, *args, **kwargs):
        f = lambda A: A.real.view(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.view(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.view(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)
    
    def reshape(self, *args, **kwargs):
        f = lambda A: A.real.reshape(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.reshape(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.reshape(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)
    
    def transpose(self, *args, **kwargs):
        f = lambda A: A.real.transpose(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.transpose(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.transpose(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def movedim(self, *args, **kwargs):
        f = lambda A: A.real.movedim(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.movedim(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.movedim(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)
    
    def split(self, *args, **kwargs):
        f = lambda A: A.real.split(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.split(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.split(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def repeat(self, *args, **kwargs):
        f = lambda A: A.real.repeat(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.repeat(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.repeat(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)
    
    def gather(self, *args, **kwargs):
        f = lambda A: A.real.gather(*args, **kwargs)
        fa = lambda real, A, A_eps1: A_eps1.gather(*args, **kwargs)
        faa = lambda real, A:  A.eps1eps2.gather(*args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)

    def __getitem__(self, key):
        f = lambda A: A.real[key]
        fa = lambda real, A, A_eps1: A_eps1[key]
        def faa(real, A):
            if A.eps1eps2 is not None:
                return A.eps1eps2[key]
            else:
                return None
        return HyperTensor.unary(self, f, fa, faa)

    def __setitem__(self, key, value):
        # if isinstance(value, (int, float, torch.Tensor)):
        #     value = HyperTensor(value)
        def f(A):
            A.real[key] = value.real
            return A.real
        def fa(real, A, A_eps1):
            A_eps1[key] = value.eps1
            return A_eps1
        def faa(real, A):
            if A.eps1eps2 is not None:
                A.eps1eps2[key] = value.eps1eps2
                return A.eps1eps2
            else:
                return None
        return HyperTensor.unary(self, f, fa, faa)
        
    def pow(self, exponent):
        if isinstance(exponent, (int, float, torch.Tensor)):
            # Handle scalar multiplication
            other = HyperTensor(exponent)
        f = lambda A, B: A.real ** B.real
        fa = lambda real, A, B, A_eps1, B_eps1: B.real * A.real ** (B.real - 1) * A_eps1
        fb = lambda real, A, B, A_eps1, B_eps1: A.real ** B.real * torch.log(A.real) * B_eps1
        f1all = lambda real, A, B, A_eps1, B_eps1: A.real ** (B.real - 1) * (B.real * A_eps1 + A.real * torch.log(A.real) * B_eps1)
        faa = lambda real, A, B: B.real * A.real ** (B.real - 2) * ((0. if A.eps1eps2 is None else A.real * A.eps1eps2) + (B.real - 1) * A.eps1 * A.eps2)
        fbb = lambda real, A, B: torch.log(A.real) * A.real ** B.real * (torch.log(A.real) * B.eps1 * B.eps2 + (0. if B.eps1eps2 is None else B.eps1eps2))
        def f2all(real, A, B):
            if A.eps1eps2 is None:
                if B.eps1eps2 is None:
                    return A.real**(B.real - 2) * ( 
                                A_real * (A.eps1 * B.eps2 + A.eps2 * B.eps1) + 
                                A.real * B.real * A.eps1 * np.log(A_real) * B.eps2 +
                                A.real * B.real * A.eps2 * np.log(A_real) * B.eps1 +
                                B.real**2 * A.eps1 * A.eps2 - 
                                B.real * A.eps1 * A.eps2 + 
                                A.real**2 * np.log(A_real)**2 * B.eps1*B.eps2)
                else:
                    return A.real**(B.real - 2) * ( 
                                A_real * (A.eps1 * B.eps2 + A.eps2 * B.eps1) + 
                                A.real * B.real * A.eps1 * np.log(A_real) * B.eps2 +
                                A.real * B.real * A.eps2 * np.log(A_real) * B.eps1 +
                                B.real**2 * A.eps1 * A.eps2 - 
                                B.real * A.eps1 * A.eps2 + 
                                A.real**2 * np.log(A.real) * B.eps1eps2 + 
                                A.real**2 * np.log(A_real)**2 * B.eps1*B.eps2)
            elif B.eps1eps2 is None:
                return A.real**(B.real - 2) * (A.real * B.real * A.eps1eps2 + 
                                A_real * (A.eps1 * B.eps2 + A.eps2 * B.eps1) + 
                                A.real * B.real * A.eps1 * np.log(A_real) * B.eps2 +
                                A.real * B.real * A.eps2 * np.log(A_real) * B.eps1 +
                                B.real**2 * A.eps1 * A.eps2 - 
                                B.real * A.eps1 * A.eps2 +  
                                A.real**2 * np.log(A_real)**2 * B.eps1*B.eps2)
            else:
                return A.real**(B.real - 2) * (A.real * B.real * A.eps1eps2 + 
                                A_real * (A.eps1 * B.eps2 + A.eps2 * B.eps1) + 
                                A.real * B.real * A.eps1 * np.log(A_real) * B.eps2 +
                                A.real * B.real * A.eps2 * np.log(A_real) * B.eps1 +
                                B.real**2 * A.eps1 * A.eps2 - 
                                B.real * A.eps1 * A.eps2 + 
                                A.real**2 * np.log(A.real) * B.eps1eps2 + 
                                A.real**2 * np.log(A_real)**2 * B.eps1*B.eps2)
        return HyperTensor.binary(self, exponent, f, fa, fb, f1all, faa, fbb, f2all)

    def __pow__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            other = HyperTensor(other)
        return self.pow(other)
    
    def __rpow__(self, other):
        if isinstance(other, (int, float, torch.Tensor)):
            other = HyperTensor(other)
        return other.pow(self)
    
    def conv2d(self, other, *args, **kwargs):
        if isinstance(other, (int, float, torch.Tensor)):
            # Handle scalar multiplication
            other = HyperTensor(other)
        f = lambda A, B: torch.conv2d(A.real, B.real, *args, **kwargs)
        fa = lambda real, A, B, A_eps1, B_eps1: torch.conv2d(A_eps1, B.real, *args, **kwargs)
        fb = lambda real, A, B, A_eps1, B_eps1: torch.conv2d(A.real, B_eps1, *args, **kwargs)
        f1all = lambda real, A, B, A_eps1, B_eps1: torch.conv2d(A_eps1, B.real, *args, **kwargs) + torch.conv2d(A.real, B_eps1, *args, **kwargs) 
        faa = lambda real, A, B: torch.conv2d(A.eps1eps2, B.real, *args, **kwargs) if A.eps1eps2 is not None else torch.zeros_like(real)
        fbb = lambda real, A, B: torch.conv2d(A.real, B.eps1eps2, *args, **kwargs) if B.eps1eps2 is not None else torch.zeros_like(real)
        def f2all(real, A, B):
            if A.eps1eps2 is None:
                if B.eps1eps2 is None:
                    return torch.conv2d(A.eps1, B.eps2, *args, **kwargs) + torch.conv2d(A.eps2, B.eps1, *args, **kwargs)
                else:
                    return torch.conv2d(A.eps1, B.eps2, *args, **kwargs) + torch.conv2d(A.eps2, B.eps1, *args, **kwargs) + torch.conv2d(A.real, B.eps1eps2, *args, **kwargs)
            elif B.eps1eps2 is None:
                return torch.conv2d(A.eps1, B.eps2, *args, **kwargs) + torch.conv2d(A.eps2, B.eps1, *args, **kwargs) + torch.conv2d(A.eps1eps2, B.real, *args, **kwargs)
            else:
                return torch.conv2d(A.eps1, B.eps2, *args, **kwargs) + torch.conv2d(A.eps2, B.eps1, *args, **kwargs) + torch.conv2d(A.eps1eps2, B.real, *args, **kwargs) + torch.conv2d(A.real, B.eps1eps2, *args, **kwargs) 
        return HyperTensor.binary(self, other, f, fa, fb, f1all, faa, fbb, f2all)
    
    def maxpool2d(self, *args, **kwargs):
        if self.real.dim() != 4:
            raise ValueError('Expecting a 4d tensor with shape BxCxHxW')
        res, indices = torch.nn.functional.max_pool2d_with_indices(self.real, *args, **kwargs)
        f = lambda A: res
        fa = lambda real, A, A_eps1: A_eps1.flatten(start_dim=2).gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices)
        faa = lambda real, A:  A.eps1eps2.flatten(start_dim=2).gather(dim=2, index=indices.flatten(start_dim=2)).view_as(indices) if A.eps1eps2 is not None else torch.zeros_like(res)
        return HyperTensor.unary(self, f, fa, faa)
    
    def adaptiveavgpool2d(self, *args, **kwargs):
        f = lambda A: torch.nn.functional.adaptive_avg_pool2d(A.real, *args, **kwargs)
        fa = lambda real, A, A_eps1: torch.nn.functional.adaptive_avg_pool2d(A_eps1, *args, **kwargs)
        faa = lambda real, A:  torch.nn.functional.adaptive_avg_pool2d(A.eps1eps2, *args, **kwargs) if A.eps1eps2 is not None else None
        return HyperTensor.unary(self, f, fa, faa)
    
    # Composed functions:
    def logsumexp(self, dim=0, epsilon = 1e-12):
        # Composed function
        amax = self.real.max(dim, keepdims=True)[0]
        e = (self - amax).exp()
        return amax.sum(dim) + (e.sum(dim) + epsilon).log()

    def logsoftmax(self, dim=0):
        # Composed function
        return self - self.logsumexp(dim).unsqueeze(dim)

    def sigmoid(self):
        # Composed function
        return 1. / (1. + (-self).exp())
    
    def softmax(self, dim=0, epsilon = 1e-12):
        amax = self.real.max(dim, keepdims=True)[0]
        e = (self - amax).exp()
        return e / (e.sum(dim, keepdims=True) + epsilon)
