import unittest
import torch
import torch.nn as nn
from fomoh.util import hess_vector_build, jac_vector_build
from fomoh.hyperdual import HyperTensor as htorch
from fomoh.nn import LogisticRegressionModel, nll_loss, CNNModel, DenseModel
from fomoh.opt import flatten_loss_model_batched, flatten_loss_model_batched_vectorized
import timeit

gpu = 0

class TestTensorEquality(unittest.TestCase):
    def test_rosenbrock(self):
        """
        Test known Hessian and HVP with rosenbrock function
        """
        def rosenbrock(x):
            return (1.-x[:,0]) ** 2. + 100. * (x[:,1] - x[:,0]**2)**2

        def rosenbrockJacobian(x):
            return torch.tensor([[-2.*(1-x[:,0])-400.*x[:,0]*(x[:,1] - x[:,0]**2)], [200.*(x[:,1] - x[:,0]**2)]])

        def rosenbrockHessian(x):
            return torch.tensor([[2.+1200.*x[:,0]*x[:,0]-400.*x[:,1], -400.*x[:,0]],[-400.*x[:,0], 200.*1.]])
        
        x = torch.randn(2).view(1,-1)

        H = torch.zeros(2,2)
        J1 = torch.zeros(1,2)
        J2 = torch.zeros(1,2)

        for tensors in hess_vector_build(2):
            xeps1 = tensors[0].view(1,-1)
            xeps2 = tensors[1].view(1,-1)
            x_htorch = htorch(x, xeps1, xeps2)
            z = rosenbrock(x_htorch)
            H[(tensors[0].view(-1,1) @ tensors[1].view(1,-1)).bool()] += z.eps1eps2.sum()
            J1[xeps1.bool()] += z.eps1.sum()
            J2[xeps2.bool()] += z.eps2.sum()
        
        self.assertTrue(torch.allclose(H, rosenbrockHessian(x)), "Rosenbrock Hessians do not match")
        
        x = torch.randn(2).view(1,-1)
        v = torch.randn(2).view(1,-1)

        Hv = torch.zeros(1,2)
        J1 = torch.zeros(1,2)

        i = 0
        for tensors in jac_vector_build(2):
            xeps1 = tensors.view(1,-1)
            xeps2 = v.clone()
            x_htorch = htorch(x, xeps1, xeps2)
            z = rosenbrock(x_htorch)
            Hv[xeps1.bool()] += z.eps1eps2.sum()
            J1[xeps1.bool()] += z.eps1.sum()
            i += 1
        
        self.assertTrue(torch.allclose(Hv, v @ rosenbrockHessian(x)), "Rosenbrock Hessian-vector products do not match")
        self.assertTrue(torch.allclose(J1, rosenbrockJacobian(x).t()), "Rosenbrock Jacobians do not match")

    def test_check_backwards_through_log_reg(self):
        
        class LogRegTorch(nn.Module):
            def __init__(self):
                super(LogRegTorch, self).__init__()
                self.fc1 = nn.Linear(28*28,10)

            def forward(self, x):
                x = self.fc1(x)
                return torch.log_softmax(x, -1)
        
        data = torch.randn(128, 1, 28, 28)
        label = torch.ones(128).long()
        
        model_torch = LogRegTorch()
        model = LogisticRegressionModel(28*28, 10)

        model.nn_module_to_htorch_model(model_torch)
        out = model_torch(data.view(-1,28*28))
        hout = model(htorch(data.view(-1,28*28)), None)

        self.assertTrue(torch.allclose(out, hout.real, rtol=1e-04, atol=1e-04), "Logistic Regression models do not match.")
        
        # Backprop through an HTorch model
        # Zero grad manually
        for p in model.params.values():
            p.grad = None
        loss_module = lambda x, y: nll_loss(x, y)
        v = torch.randn(model.n_params)
        pred = model(htorch(data.view(-1,28*28)), model.vec_to_params(v), requires_grad=True)
        z = loss_module(pred, htorch(label))
        z.real.backward()
        param_directions = []
        for p in model.params.values():
            param_directions.append(p.grad)
        param_directions = model.params_to_vec(param_directions)
        
        # Backprop through a nn.Module
        for p in model_torch.parameters():
            p.grad = None

        crit = torch.nn.CrossEntropyLoss()
        loss_module_t = lambda x, y: crit(x, y)
        pred = model_torch(data.view(-1,28*28))
        out = loss_module_t(pred, label)
        out.backward()
        grads = model.collect_nn_module_grads(model_torch)
        grads = model.params_to_vec(grads)

        self.assertTrue(torch.allclose(param_directions, grads, rtol=1e-5,atol=1e-5), f"Logistic Regression models backprop do not match. {abs(param_directions - grads).max()}")
        self.assertTrue(torch.allclose(grads @ v.t(), z.eps1), f"Logistic Regression models directional derivatives do not match. {abs(grads @ v.t() - z.eps1).max()}")
        
        # Check HVP
        pred = model_torch(data.view(-1,28*28))
        loss = loss_module_t(pred, label)

        # First gradient calculation
        grads = torch.autograd.grad(loss, model_torch.parameters(), create_graph=True)

        # Assume v is some vector. In practice, you would define this.
        v_dash = model.vec_to_params(v)
        v_dash = [v_.t() if len(v_.shape) == 2 else v_ for v_ in v_dash]

        # Compute the Hessian-vector product (Hv)
        Hv = torch.autograd.grad(grads, model_torch.parameters(), grad_outputs=v_dash, only_inputs=True)

        # Calculate v^T H v by dotting 'v' with 'Hv'
        vTHv = sum((v_.flatten().dot(hv.flatten().detach()) for v_, hv in zip(v_dash, Hv)))
        
        self.assertTrue(torch.allclose(vTHv, z.eps1eps2), f"Logistic Regression models curvature does not match. {abs(vTHv - z.eps1eps2).max()}")
        
    def test_vectorized_DenseModel(self):     
        """
        Test the sequential batched version vs. vectorized version, that they are the same for the Dense Model.
        """

        data = torch.randn(16, 1, 28, 28)
        label = torch.randint(0,10, (16,)).long()
        loss_module = lambda x, y: nll_loss(x.logsoftmax(-1), y)

        model = DenseModel([28*28, 100, 100, 100, 10]) #LogisticRegressionModel(28*28, 10)
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        model.to(device)

        fun = flatten_loss_model_batched(model, loss_module, data.view(-1,28*28).to(device), label.to(device))

        fun_vect = flatten_loss_model_batched_vectorized(model, loss_module, data.view(-1,28*28).to(device), label.to(device))

        N = 50

        V1 = []
        V2 = []

        D = int((N**2 + N)/2)

        x_repeat = model.params_to_vec().repeat(D,1)
        tangents = torch.randn(N, x_repeat.shape[1]).to(x_repeat.device)

        for i in range(N):
            for j in range(i, N):
                V1.append(tangents[i])
                V2.append(tangents[j])

        x_htorch = htorch(x_repeat, torch.stack(V1), torch.stack(V2))

        _, out_seq = fun(x_htorch)
        _, out_par = fun_vect(x_htorch)

        self.assertTrue(torch.allclose(out_seq.real, out_par.real, rtol=1e-04, atol=1e-02), f"Dense sequential does not match vectorized. Abs: {abs(out_seq.real- out_par.real).max()}, Rel: {abs(out_par.real).max()}")
        self.assertTrue(torch.allclose(out_seq.eps1, out_par.eps1, rtol=1e-04, atol=1e-02),f"Dense sequential does not match vectorized. Abs: {abs(out_seq.eps1- out_par.eps1).max()}, Rel: {abs(out_par.eps1).max()}")
        self.assertTrue(torch.allclose(out_seq.eps2, out_par.eps2, rtol=1e-04, atol=1e-02),f"Dense sequential does not match vectorized. Abs: {abs(out_seq.eps2- out_par.eps2).max()}, Rel: {abs(out_par.eps2).max()}")
        self.assertTrue(torch.allclose(out_seq.eps1eps2, out_par.eps1eps2, rtol=1e-04, atol=1e-02),f"Dense sequential does not match vectorized. Abs: {abs(out_seq.eps1eps2- out_par.eps1eps2).max()}, Rel: {abs(out_par.eps1eps2).max()}")


        # Time the function running 10 times
        execution_time = timeit.timeit(lambda: fun(x_htorch), globals=globals(), number=5)
        execution_time_vec = timeit.timeit(lambda: fun_vect(x_htorch), globals=globals(), number=5)
        print("On device: ", device)
        print("Average Seqential DenseNet Time:", execution_time / 5, "seconds")
        print("Average Vectorize DenseNet Time:", execution_time_vec / 5, "seconds")
        
    
    def test_vectorized_CNNModel(self):     
        """
        Test the sequential batched version vs. vectorized version, that they are the same for the Dense Model.
        """

        data = torch.randn(16, 1, 28, 28)
        label = torch.randint(0,10, (16,)).long()
        loss_module = lambda x, y: nll_loss(x.logsoftmax(-1), y)

        cnn_kwargs = {'cnn_layers_channels': [1,20,50],
        'cnn_filter_size': 5,
        'dense_layers': [4*4*50, 500, 10],
        'maxpool_args': [2,2],
        'bias': True}
        model = CNNModel(**cnn_kwargs)
        device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
        model.to(device)

        fun = flatten_loss_model_batched(model, loss_module, data.to(device), label.to(device))

        fun_vect = flatten_loss_model_batched_vectorized(model, loss_module, data.to(device), label.to(device))

        N = 10

        V1 = []
        V2 = []

        D = int((N**2 + N)/2)

        x_repeat = model.params_to_vec().repeat(D,1)
        tangents = torch.randn(N, x_repeat.shape[1]).to(x_repeat.device)

        for i in range(N):
            for j in range(i, N):
                V1.append(tangents[i])
                V2.append(tangents[j])

        x_htorch = htorch(x_repeat, torch.stack(V1), torch.stack(V2))

        _, out_seq = fun(x_htorch)
        _, out_par = fun_vect(x_htorch)

        self.assertTrue(torch.allclose(out_seq.real, out_par.real, rtol=1e-03, atol=1e-03), f"CNN sequential does not match vectorized. Abs: {abs(out_seq.real- out_par.real).max()}, Rel: {abs(out_par.real).max()}")
        self.assertTrue(torch.allclose(out_seq.eps1, out_par.eps1, rtol=1e-03, atol=1e-03),f"CNN sequential does not match vectorized. Abs: {abs(out_seq.eps1- out_par.eps1).max()}, Rel: {abs(out_par.eps1).max()}")
        self.assertTrue(torch.allclose(out_seq.eps2, out_par.eps2, rtol=1e-03, atol=1e-03),f"CNN sequential does not match vectorized. Abs: {abs(out_seq.eps2- out_par.eps2).max()}, Rel: {abs(out_par.eps2).max()}")
        self.assertTrue(torch.allclose(out_seq.eps1eps2, out_par.eps1eps2, rtol=1e-03, atol=1e-03),f"CNN sequential does not match vectorized. Abs: {abs(out_seq.eps1eps2- out_par.eps1eps2).max()}, Rel: {abs(out_par.eps1eps2).max()}")


        # Time the function running 10 times
        execution_time = timeit.timeit(lambda: fun(x_htorch), globals=globals(), number=5)
        execution_time_vec = timeit.timeit(lambda: fun_vect(x_htorch), globals=globals(), number=5)
        print("On device: ", device)
        print("CNN: Average Seqential Time:", execution_time / 5, "seconds")
        print("CNN: Average Vectorize Time:", execution_time_vec / 5, "seconds")


if __name__ == '__main__':
    unittest.main()