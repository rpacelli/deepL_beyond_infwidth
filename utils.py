from __future__ import print_function
import os, os.path,time,argparse
import numpy as np
import torch, torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.optim as optim

def parseArguments():
    parser = argparse.ArgumentParser()
    # Positional mandatory arguments
    parser.add_argument("teacher_type", help="choose a teacher type: linear,random, mnist", type=str)
    
    # Net arguments
    parser.add_argument("-L", help="number of hidden layers", type=int, default=1)
    parser.add_argument("-N", "--N", help="size of input data", type=int, default=400)
    parser.add_argument("-K", "--K", help="number of filters", type=int, default=5) 
    parser.add_argument("-f", "--f", help="size of filter", type=int, default=3) 
    parser.add_argument("-stride", "--stride", help="stride", type=int, default=1)      
    parser.add_argument("-act", help="activation function", type=str, default="erf")
    # Learning dynamics arguments
    parser.add_argument("-lr", "--lr", help="learning rate", type=float, default=1e-03)
    parser.add_argument("-wd", "--wd", help="weight decay", type=float, default=0.)
    parser.add_argument("-opt", "--opt", type=str, default="sgd") #or adam
    parser.add_argument("-device", "--device",  type=str, default="cpu")
    parser.add_argument("-epochs", "--epochs", help="number of train epochs", type = int, default = 10000)
    parser.add_argument("-checkpoint", "--checkpoint", help="# epochs checkpoint", type=int, default=1000)
    parser.add_argument("-R", "--R", help="replica index", type=int, default=1)
    # Data specification
    parser.add_argument("-bs", "--bs", help="batch size", type=int, default=0)
    parser.add_argument("-P", "--P", help="size of training set", type=int, default=500)
    parser.add_argument("-Ptest", "--Ptest", help="# examples in test set", type=int, default=1000)    
    parser.add_argument("-save_data", "--save_data", type = bool, default= True)
    parser.add_argument("-resume_data", help="try resuming data from checkpoint", type=bool, default=True) 
    # Theory computation
    parser.add_argument("-compute_theory", type = bool, default= False)
    parser.add_argument("-infwidth", help="compute infinite width theory", type = bool, default= False)
    parser.add_argument("-lambda1", type = float, default= 1.)
    parser.add_argument("-lambda0", type = float, default= 1.)
    args = parser.parse_args()
    return args

def train_prep(net, data, labels, batch_size):
    net.train()
    P = len(data)
    batch_num = max(int(P/batch_size),1)
    s = np.arange(P)
    np.random.shuffle(s)
    data = data[s]
    labels = labels[s]
    return data, labels, batch_num

def train(net,data, labels, criterion, optimizer,batch_size,norm):
    data, labels, batch_num = train_prep(net, data, labels, batch_size)
    train_loss = 0
    for i in range(batch_num):
        start = i*(batch_size)
        inputs, targets = data[start:start+batch_size], (labels[start:start+batch_size])#.unsqueeze(1)
        optimizer.zero_grad()
        outputs = norm*net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() 
    return train_loss/batch_num

def test(net, test_data, test_labels, criterion,  batch_size,norm):
        net.eval()
        test_loss = 0
        P_test = len(test_data)
        batch_num = max(int(P_test/batch_size),1)
        for i in range(batch_num):
            start = i*(batch_size)
            with torch.no_grad():
                    #inputs, targets = test_data[start:start+batch_size], (test_labels[start:start+batch_size])#.unsqueeze(1)
                    #outputs = net(inputs)
                    #loss = criterion(outputs, targets) 
                    #test_loss += loss.item()
                    inputs, targets = test_data[start:start+batch_size], (test_labels[start:start+batch_size])#.unsqueeze(1)
                    outputs = norm*net(inputs)
                    signs = outputs*targets
                    signs /= abs(signs)
                    signs +=1
                    loss = torch.count_nonzero(signs)/batch_size
                    print("loss is ", loss)
                    #loss = criterion(outputs, targets) 
                    test_loss += loss.item()                    
        return test_loss/batch_num



class Erf(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.erf(x)


class ConvNet:
    def __init__(self, K, f, L, lambda1, lambda0):
        self.K = K
        self.f = f
        self.L = L
        self.lambda0 = lambda0
        self.lambda1 = lambda1

    def Sequential(self, input_channels, input_size, stride, bias, act_func):
        if act_func == "relu":
            act = nn.ReLU()
        elif act_func == "erf":
            act = Erf()
        std_0 = 1/np.sqrt(self.lambda0)
        std_1 = 1/np.sqrt(self.lambda1)

        modules = []

        # First convolutional layer
        first_layer = nn.Conv2d(input_channels, self.K, self.f, bias=bias)
        init.normal_(first_layer.weight, std=std_0)
       # init.normal_(first_layer.weight, std=std_0/np.sqrt(input_channels * self.f * self.f))
        if bias:
            init.constant_(first_layer.bias, 0)
        modules.extend([first_layer, act])

        # Additional L-1 convolutional layers
        for l in range(self.L - 1):
            layer = nn.Conv2d(self.K, self.K, self.f, bias=bias)
            init.normal_(layer.weight, std=std_1)
            #init.normal_(layer.weight, std=std_1/np.sqrt(self.K * self.f * self.f))
            if bias:
                init.constant_(layer.bias, 0)
            modules.extend([layer, act])
        # Flatten the tensor before the fully connected layer
        modules.append(nn.Flatten())

        # Calculate the output size of the last convolutional layer to determine the number of input features for the fully connected layer
        # Assume no padding 
        conv_output_size = input_size
        for _ in range(self.L):
            conv_output_size = (conv_output_size - self.f) // stride + 1

        # Print the number of parameters in the fully connected layer
        FC_params = self.K * conv_output_size * conv_output_size
        print(f"Number of parameters in the fully connected layer: {FC_params * input_channels}")

        # Fully connected layer
        last_layer = nn.Linear(FC_params, 1, bias=bias)
        init.normal_(last_layer.weight, std=std_1)
        if bias:
            init.constant_(last_layer.bias, 0)
        modules.append(last_layer)

        sequential = nn.Sequential(*modules)
        print(f'\nThe network has {self.L} convolutional hidden layer(s) with {self.K} kernels of size {self.f} and {act_func} activation function', sequential)
        return sequential, 1/np.sqrt(FC_params*input_size)


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

def cuda_init(net, device):
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        pin_memory=False
        #cudnn.benchmark = True
        #CUDA_LAUNCH_BLOCKING=1

def make_directory(dir_name):
    if not os.path.isdir(dir_name): 
	    os.mkdir(dir_name)

def make_folders(mother_dir, args):
    #CREATION OF 2ND FOLDER WITH TEACHER & NETWORK SPECIFICATIONS
    first_subdir = mother_dir + f'teacher_{args.teacher_type}_net_{args.L}hl_opt_{args.opt}_actf_{args.act}/'
    make_directory(first_subdir)
    #CREATION OF 3RD FOLDER WITH RUN SPECIFICATIONS
    attributes_string = f'lr_{args.lr}_w_decay_{args.wd}_lambda0_{args.lambda0}_lambda1_{args.lambda1}_'
    if args.bs != 0: 
        attributes_string = attributes_string +f'bs_{args.bs}_'
    attributes_string = attributes_string + f"{args.L}hl_N0_{args.N}_N_{args.K}"
    run_folder = first_subdir + attributes_string + '/'
    make_directory(run_folder)
    return first_subdir, run_folder

def k0(x,y,lambda0):
    N0 = len(x)
    return (1/(lambda0*N0)) * np.dot(x,y)

def CorrMat(data,lambda0):
    P = len(data)
    C = np.zeros((P,P))
    for i in range(P): 
        for j in range(i,P):         
            C[i][j] = k0(data[i].flatten(), data[j].flatten(), lambda0)
            C[j][i] = C[i][j]
    return C  

def Conv_ker(data,norm):
    P = len(data)
    C = np.zeros((P,P))
    for i in range(P): 
        for j in range(i,P):         
            C[i][j] = torch.mean(data[i]*data[j])/norm
            C[j][i] = C[i][j]
    return C  

def kmatrix(C,kernel,lambda1):
    P = len(C)
    K = np.zeros((P,P))
    for i in range(P): 
        for j in range(i,P):         
            K[i][j] = kernel(C[i][i], C[i][j], C[j][j],lambda1)
            K[j][i] = K[i][j]
    return K

def kernel_erf(k0xx,k0xy,k0yy,lambda1):
    return (2/(lambda1*np.pi))*np.arcsin((2*k0xy)/np.sqrt((1+2*k0xx)*(1+2*k0yy)))