import torch
import wandb
from fomoh.hyperdual import HyperTensor as htorch
from fomoh.nn import CNNModel, nll_loss, LogisticRegressionModel, optimizer_step, CNN_CIFAR10
from fomoh.opt import optimizer_step_plane_Nd
from fomoh.nn_models_torch import CNN_CIFAR10_torch
import time
import datetime
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np


def get_time_stamp():
    return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d_%H%M%S')

class LogRegTorch(torch.nn.Module):
    def __init__(self):
        super(LogRegTorch, self).__init__()
        self.fc1 = torch.nn.Linear(28*28,10)
        
    def forward(self, x):
        x = self.fc1(x)
        return torch.log_softmax(x, -1)
    
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.CW1 = torch.nn.Conv2d(1, 20, 5)
        self.CW2 = torch.nn.Conv2d(20, 50, 5)
        self.fc1 = torch.nn.Linear(4*4*50,500)
        self.fc2 = torch.nn.Linear(500, 10)
        self.maxpool = torch.nn.MaxPool2d(2,2)
        
    def forward(self, x):
        x = F.relu(self.CW1(x)) # relu needs to be done at some point
        # print(x.shape)
        x = self.maxpool(x)
        # print(x.shape)
        x = F.relu(self.CW2(x)) # relu needs to be done at some point
        x = self.maxpool(x)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def logistic_regression():
    model = LogisticRegressionModel(28*28, 10)
    model_torch = LogRegTorch()
    model.nn_module_to_htorch_model(model_torch)
    loss_module = lambda x, y: nll_loss(x, y)
    crit = torch.nn.NLLLoss()
    loss_module_torch = lambda x, y: crit(x, y)
    return model, model_torch, loss_module, loss_module_torch

def cnn_mnist():
    cnn_kwargs = {'cnn_layers_channels': [1,20,50],
                    'cnn_filter_size': 5,
                    'dense_layers': [4*4*50, 500, 10],
                    'maxpool_args': [2,2],
                    'bias': True}
    model = CNNModel(**cnn_kwargs)
    model_torch = CNN()
    model.nn_module_to_htorch_model(model_torch)
    loss_module = lambda x, y: nll_loss(x.logsoftmax(-1), y)
    crit = torch.nn.CrossEntropyLoss()
    loss_module_torch = lambda x, y: crit(x, y)
    return model, model_torch, loss_module, loss_module_torch

def cnn_cifar10():
    model = CNN_CIFAR10(dropout=True)
    model_torch = CNN_CIFAR10_torch()
    model.nn_module_to_htorch_model(model_torch)
    loss_module = lambda x, y: nll_loss(x.logsoftmax(-1), y)
    crit = torch.nn.CrossEntropyLoss()
    loss_module_torch = lambda x, y: crit(x, y)
    return model, model_torch, loss_module, loss_module_torch

def make_data(modelClass):
    if modelClass == "logreg" or modelClass == "cnn":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(root='/tmp/mnist', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(root='/tmp/mnist', train=False, download=True, transform=transform)
    elif modelClass == "cnn-cifar10":
        transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

        train_dataset = datasets.CIFAR10("/tmp/cifar10", train=True, download=True,
                                transform=transform_train)
        test_dataset = datasets.CIFAR10("/tmp/cifar10", train=False, download=True,
                                transform=transform_test)
    else:
        raise NotImplementedError
    
    return train_dataset, test_dataset

def val_step(test_loader, model, loss_module, epoch, loss_type, shape, device):
    val_loss = 0
    correct = 0
    model.eval()
    for inputs, labels in test_loader:
        inputs = htorch(inputs.to(device)).view(-1, *shape)
        labels = htorch(labels.to(device))
        pred = model(inputs, None)
        l = loss_module(pred,labels)
        val_loss += l.real.cpu().item() * labels.real.shape[0]
        if loss_type == "classification":
            correct += sum(pred.exp().real.argmax(1).cpu() == labels.real.cpu())
    return correct/len(test_loader.dataset), val_loss/len(test_loader.dataset)
            
def FoMoH_gradient_descent(modelClass, epochs, n_sample_directions, lr, clip_value, hess, weight_decay, optimizer, train_loader, val_loader, epsilon, device, use_wandb, backprop, save, run_to_end, lr_sched, milestones, gamma = 0.1):
    
    if modelClass == "logreg":
        loss_type = "classification"
        model, _, loss_module, _ = logistic_regression()
        shape = [(torch.tensor(train_loader.dataset.data[0].shape).prod())]
        model.to(device)
    elif modelClass == "cnn":
        loss_type = "classification"
        model, _, loss_module, _ = cnn_mnist()
        shape = torch.tensor(train_loader.dataset.data[0][None].shape)
        model.to(device)
    elif modelClass == "cnn-cifar10":
        loss_type = "classification"
        model, _, loss_module, _ = cnn_cifar10()
        shape = [3, 32, 32]
        model.to(device)
    else:
        raise NotImplementedError
    
    save_dict = {'accuracy':[], 'loss':[], 'loss_valid':[], 'accuracy_valid':[]}
    
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.params.values(), lr=lr, weight_decay=weight_decay)
    if lr_sched:
        milestone_list = [milestones] * int(epochs/milestones)
        # gamma = 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_list, gamma=gamma)
    
    loss_valid_list = []
    for i in range(1, epochs+1):
        log = {}
        loss = 0
        curvature = 0
        if loss_type == "classification":
            correct = 0
        for inputs, labels in train_loader:
            # import pdb; pdb.set_trace()
            model.train()
            inputs = inputs.to(device).view(-1,*shape)
            labels = labels.to(device)
            ls, pred = optimizer_step(model, loss_module, optimizer, n_sample_directions, htorch(inputs), htorch(labels), device, clip_value=clip_value, hess = hess, epsilon=epsilon, backprop = backprop)
            curvature += inputs.shape[0] * ls.eps1eps2.item()
            loss += inputs.shape[0] * ls.real.item()
            if loss_type == "classification":
                correct += sum(torch.softmax(pred, -1).argmax(1).cpu() == labels.real.cpu())
                        
        loss = loss/len(train_loader.dataset)
        if loss_type == "classification":
            accuracy = correct/len(train_loader.dataset)
            log['accuracy'] = accuracy
        log['epoch'] = i
        log['loss'] = loss
        log['curvature'] = curvature/len(train_loader.dataset)
        
        accuracy_val, loss_val = val_step(val_loader, model, loss_module, i, loss_type, shape, device)
        
        loss_valid_list.append(loss_val)
        log['loss_valid'] = loss_val
        log['accuracy_valid'] = accuracy_val
        # print(min(loss_valid_list))
        if np.any(np.isnan(min(loss_valid_list))):
            log['best_loss_valid'] = 2.4 # approximately random choice
            if not run_to_end:
                break
        else:
            log['best_loss_valid'] = min(loss_valid_list)
        
        if i % 10 == 0:
            print('{:}: Valid loss: {:.2e} | accuracy: {:,.3f}'.format(i, loss_val, accuracy_val))
        if use_wandb:
            wandb.log(log)
        if save:
            save_dict['accuracy'].append(accuracy)
            save_dict['loss'].append(loss)
            save_dict['loss_valid'].append(loss_val)
            save_dict['accuracy_valid'].append(accuracy_val)

        if not run_to_end:
            if i >= 10 and loss_val >= 10e7:
                break
            if i >= 100 and accuracy_val < 0.4:
                # too slow so break
                break
        
        if lr_sched:
            scheduler.step()

    return save_dict
            
def FoMoH_Kd_gradient_descent(modelClass, epochs, n_sample_directions, lr, clip_value, train_loader, val_loader, device, use_wandb, save, run_to_end, K=2, lr_sched = False, milestones=1000, gamma = 0.1):
    
    if modelClass == "logreg":
        loss_type = "classification"
        model, _, loss_module, _ = logistic_regression()
        shape = [(torch.tensor(train_loader.dataset.data[0].shape).prod())]
        model.to(device)
    elif modelClass == "cnn":
        loss_type = "classification"
        model, _, loss_module, _ = cnn_mnist()
        shape = torch.tensor(train_loader.dataset.data[0][None].shape)
        model.to(device)
    elif modelClass == "cnn-cifar10":
        loss_type = "classification"
        model, _, loss_module, _ = cnn_cifar10()
        shape = [3, 32, 32]
        model.to(device)
    else:
        raise NotImplementedError
    save_dict = {'accuracy':[], 'loss':[], 'loss_valid':[], 'accuracy_valid':[]}
    loss_valid_list = []
    for i in range(1, epochs+1):
        log = {}
        loss = 0
        curvature = 0
        if loss_type == "classification":
            correct = 0
        for inputs, labels in train_loader:
            model.train()
            inputs = inputs.to(device).view(-1,*shape)
            labels = labels.to(device)
            ls, pred = optimizer_step_plane_Nd(model, loss_module, n_sample_directions, inputs, labels, device=device, clip_value=clip_value, lr=lr, N=K)
            # curvature += inputs.shape[0] * ls.eps1eps2.item()
            loss += inputs.shape[0] * ls.real.item()
            if loss_type == "classification":
                correct += sum(torch.softmax(pred, -1).argmax(1).cpu() == labels.real.cpu())
                        
        loss = loss/len(train_loader.dataset)
        if loss_type == "classification":
            accuracy = correct/len(train_loader.dataset)
            log['accuracy'] = accuracy
        log['epoch'] = i
        log['loss'] = loss
        # log['curvature'] = curvature/len(train_loader.dataset)
        
        accuracy_val, loss_val = val_step(val_loader, model, loss_module, i, loss_type, shape, device)
        
        loss_valid_list.append(loss_val)
        log['loss_valid'] = loss_val
        log['accuracy_valid'] = accuracy_val
        if i > 1:
            if np.any(np.isnan(min(loss_valid_list))):
                log['best_loss_valid'] = 2.4 # approximately random choice
                if not run_to_end:
                    break
            else:
                log['best_loss_valid'] = min(loss_valid_list)
        
        if i % 10 == 0:
            print('{:}: Valid loss: {:.2e} | accuracy: {:,.3f}'.format(i, loss_val, accuracy_val))
        if use_wandb:
            wandb.log(log)
        
        if save:
            save_dict['accuracy'].append(accuracy)
            save_dict['loss'].append(loss)
            save_dict['loss_valid'].append(loss_val)
            save_dict['accuracy_valid'].append(accuracy_val)

        if not run_to_end:
            if i >= 10 and loss_val >= 10e7:
                break
            if i >= 100 and accuracy_val < 0.4:
                # too slow so break
                break

        if lr_sched:
            # gamma = 0.1
            
            if i > 0 and i % milestones == 0:
                lr *= gamma
                print(f"learning rate reduction: lr = {lr}")
            
    return save_dict
            

def backprop_step(model, loss_module, optimizer, inputs, labels, device, clip_value=None):

    # Ensure optimizer gradients are zeroed out initially to prevent accumulation from previous iterations
    optimizer.zero_grad()

    # Forward pass: compute predicted outputs by passing inputs to the model
    outputs = model(inputs)

    # Calculate the loss based on the model's output and the true labels
    loss = loss_module(outputs, labels) 
    
    # Backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()

    # If a clip_value is specified, clip the gradients to mitigate exploding gradients problem
    if clip_value != 0.0:
        # Clip gradients to within [-clip_value, clip_value]
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-clip_value, clip_value)

    # Perform a single optimization step (i.e., parameter update)
    optimizer.step()

    # Optionally, return the loss for monitoring
    
    return loss.item(), outputs
 
def val_step_reverse(test_loader, model, loss_module, shape, device):
    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(device).view(-1, *shape)
            labels = labels.to(device)
            pred = model(inputs)
            l = loss_module(pred,labels)
            val_loss += l.cpu().item() * labels.shape[0]
            correct += sum(torch.softmax(pred, -1).argmax(1).cpu() == labels.cpu())
        return correct/len(test_loader.dataset), val_loss/len(test_loader.dataset)
            
def backprop_routine(modelClass, epochs, lr, clip_value, weight_decay, optimizer, train_loader, val_loader, device, use_wandb, save, run_to_end, lr_sched, milestones, gamma = 0.1):
    
    if modelClass == "logreg":
        loss_type = "classification"
        _, model, _, loss_module = logistic_regression()
        shape = [torch.tensor(train_loader.dataset.data[0].shape).prod()]
        model.to(device)
    elif modelClass == "cnn":
        loss_type = "classification"
        _, model, _, loss_module = cnn_mnist()
        shape = torch.tensor(train_loader.dataset.data[0][None].shape)
        model.to(device)
    elif modelClass == "cnn-cifar10":
        loss_type = "classification"
        _, model, _, loss_module = cnn_cifar10()
        shape = [3, 32, 32]
        model.to(device)
    else:
        raise NotImplementedError
        
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    if lr_sched:
        milestone_list = [milestones] * int(epochs/milestones)
        # gamma = 0.1
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_list, gamma=gamma)
    save_dict = {'accuracy':[], 'loss':[], 'loss_valid':[], 'accuracy_valid':[]}
    loss_valid_list = []
    for i in range(1, epochs+1):
        log = {}
        loss = 0
        curvature = 0
        if loss_type == "classification":
            correct = 0
        model.train()
        for inputs, labels in train_loader:
            inputs = inputs.to(device).view(-1,*shape)
            labels = labels.to(device)
            ls, pred = backprop_step(model, loss_module, optimizer, inputs, labels, device, clip_value=clip_value)
            # curvature += inputs.shape[0] * ls.eps1eps2.item()
            loss += inputs.shape[0] * ls
            if loss_type == "classification":
                correct += sum(torch.softmax(pred, -1).argmax(1).cpu() == labels.cpu())
                        
        loss = loss/len(train_loader.dataset)
        if loss_type == "classification":
            accuracy = correct/len(train_loader.dataset)
            log['accuracy'] = accuracy
        log['epoch'] = i
        log['loss'] = loss
        # log['curvature'] = curvature/len(train_loader.dataset)
        
        accuracy_val, loss_val = val_step_reverse(val_loader, model, loss_module, shape, device)
        
        loss_valid_list.append(loss_val)
        log['loss_valid'] = loss_val
        log['accuracy_valid'] = accuracy_val
        if i > 1:
            log['best_loss_valid'] = min(loss_valid_list)
        
        if i % 10 == 0:
            print('{:}: Valid loss: {:.2e} | accuracy: {:,.3f}'.format(i, loss_val, accuracy_val))
        if use_wandb:
            wandb.log(log)
        
        if save:
            save_dict['accuracy'].append(accuracy)
            save_dict['loss'].append(loss)
            save_dict['loss_valid'].append(loss_val)
            save_dict['accuracy_valid'].append(accuracy_val)

        if not run_to_end:
            if i >= 10 and loss_val >= 10e7:
                break
            if i >= 100 and accuracy_val < 0.4:
                # too slow so break
                break
        if lr_sched:
            scheduler.step()
            
    return save_dict