from __future__ import print_function
import os, os.path,time
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import argparse 
import torch.nn.init as init
import torch.optim as optim

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("teacher_type", help="choose a teacher type: linear,random, mnist", type=str)
    #optional arguments
    parser.add_argument("-L", help="number of hidden layers", type=int, default=1)
    parser.add_argument("-act", help="activation function", type=str, default="erf")
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default=1e-03)
    parser.add_argument("-wd", "--wd", help="weight decay", type=float, default=0.)
    parser.add_argument("-resume_data", help="try resuming data from checkpoint", type=bool, default=False)    
    parser.add_argument("-device", "--device",  type=str, default="cuda")
    parser.add_argument("-epochs", "--epochs", help="number of train epochs", type = int, default = 10000)
    parser.add_argument("-bs", "--bs", help="batch size", type=int, default=0)
    parser.add_argument("-P", "--P", help="size of training set", type=int, default=500)
    #specify the networks you want to use 
    parser.add_argument("-N", "--N", help="size of input data", type=int, default=300)
    parser.add_argument("-N1", "--N1", help="size ofh idden layer(s)", type=int, default=400)
    parser.add_argument("-Ptest", "--Ptest", help="# examples in test set", type=int, default=1000)
    parser.add_argument("-opt", "--opt", type=str, default="sgd") #or adam
    parser.add_argument("-R", "--R", help="replica index", type=int, default=1)
    parser.add_argument("-checkpoint", "--checkpoint", help="# epochs checkpoint", type=int, default=1000)
    parser.add_argument("-save_data", "--save_data", type = bool, default= True)
    parser.add_argument("-lambda1", type = float, default= 1.)
    parser.add_argument("-lambda0", type = float, default= 1.)
    parser.add_argument("-compute_theory", type = bool, default= False)
    parser.add_argument("-infwidth", help="compute infinite width theory", type = bool, default= False)
    args = parser.parse_args()
    return args


def net_norm(net):
	norm = 0
	for l in range(len(net)):
		with torch.no_grad():
			if isinstance(l,nn.Linear):
				norm += torch.norm(net[l].weight)**2
	return norm 

def train_prep(net, data, labels, batch_size):
    net.train()
    P = len(data)
    batch_num = max(int(P/batch_size),1)
    s = np.arange(P)
    np.random.shuffle(s)
    data = data[s]
    labels = labels[s]
    return data, labels, batch_num


    



def train(net,data, labels, criterion, optimizer,batch_size):
    data, labels, batch_num = train_prep(net, data, labels, batch_size)
    train_loss = 0
    for i in range(batch_num):
        start = i*(batch_size)
        inputs, targets = data[start:start+batch_size], (labels[start:start+batch_size]).unsqueeze(1)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
    return train_loss/batch_num



def test(net, test_data, test_labels, criterion,  batch_size):
        net.eval()
        test_loss = 0
        P_test = len(test_data)
        batch_num = max(int(P_test/batch_size),1)
        for i in range(batch_num):
            start = i*(batch_size)
            with torch.no_grad():
                    inputs, targets = test_data[start:start+batch_size], (test_labels[start:start+batch_size]).unsqueeze(1)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets) 
                    test_loss += loss.item()
        return test_loss/batch_num

def make_directory(dir_name):
    if not os.path.isdir(dir_name): 
	    os.mkdir(dir_name)


#@profile
def cuda_init(net, device):
	net = net.to(device)
	if device == 'cuda':
		net = torch.nn.DataParallel(net)
		cudnn.benchmark = True
		CUDA_LAUNCH_BLOCKING=1
	#print_stats()


class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.erf(x)


class make_MLP: 
    def __init__(self, N0, N1,  L,lambda1, lambda0):
        self.N0 = N0 
        self.N1 = N1
        self.L = L
        self.lambda0 = lambda0
        self.lambda1 = lambda1


    def Sequential(self, bias, act_func):
        if act_func == "relu":
            act = nn.ReLU()
        elif act_func == "erf":
            act = Erf()

        std_0 = 1/np.sqrt(self.lambda0)
        std_1 = 1/np.sqrt(self.lambda1)

        modules = []

        first_layer = nn.Linear(self.N0, self.N1, bias=bias)
        init.normal_(first_layer.weight, std = std_0/np.sqrt(self.N0))
        if bias:
            init.constant_(first_layer.bias,0)
        modules.append(first_layer)

        for l in range(self.L-1): 
            modules.append(act)
            layer = nn.Linear(self.N1, self.N1, bias = bias)
            init.normal_(layer.weight, std = std_1/np.sqrt(self.N1))
            if bias:
                init.constant_(layer.bias,0)
            modules.append(layer)

        modules.append(act)
        last_layer = nn.Linear(self.N1, 1, bias=bias)  
        init.normal_(last_layer.weight, std = std_1/np.sqrt(self.N1))  
        if bias:
                init.constant_(last_layer.bias,0)
        modules.append(last_layer)

        sequential = nn.Sequential(*modules)
        print(sequential)
        return sequential



def find_device(device):
    try:
        if device == 'cpu':
            raise TypeError
        torch.cuda.is_available() 
        device = 'cuda'	
    except:
        device ='cpu'
        print('\nWorking on', device)
    return device
