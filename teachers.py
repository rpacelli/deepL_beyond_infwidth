
import torch, torchvision, torchvision.transforms as t   
import numpy as np

def save_data(inputs,targets,test_inputs,test_targets, trainsetFilename):
    print('\nSaving data...')
    state = {
    	'inputs': inputs,
    	'targets': targets,
        'test_inputs': test_inputs,
    	'test_targets': test_targets,
    }
    torch.save(state, trainsetFilename)

class MNISTDataset:
    def __init__(self, N):
        self.N = N
        self.save_data = bool("NaN")

    def _get_transforms(self):
        T = t.Compose([
            t.Resize(size=int(np.sqrt(self.N))),
            t.ToTensor(),
            t.Normalize((0.1307,), (0.3081,)),
            #t.Lambda(lambda x: torch.flatten(x)) in case you have to flatten
        ])
        return T

    def make_data(self, P, P_test, trainsetFilename, device):
        transform_dataset = self._get_transforms()

        trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_dataset)
        trainset_small = torch.utils.data.Subset(trainset, list(range(P)))
        trainloader = torch.utils.data.DataLoader(trainset_small, batch_size=P, num_workers=4)

        testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_dataset)
        testset_small = torch.utils.data.Subset(testset, list(range(P_test)))
        testloader = torch.utils.data.DataLoader(testset_small, batch_size=P_test, num_workers=4)

        data, labels = next(iter(trainloader))
        labels = (torch.sign(labels.float() - 4.5)).view(-1, 1)
        
        test_data, test_labels = next(iter(testloader))
        test_labels = (torch.sign(test_labels.float() - 4.5)).view(-1, 1)

        return data, labels, test_data, test_labels, False

class linear_dataset: 
    def __init__(self):
        self.teacher_vec = float("NaN")
        self.save_data = bool("NaN")
        self.resume = bool("NaN")
    def make_teacher_parameters(self, N, teacherFilename):
        try:
            self.teacher_vec = torch.load(teacherFilename)
            print("\nLoading teacher")
            if not self.resume:
                raise(Exception)
        except:
            print("\nDidn't find a teacher, creating new one...")
            self.teacher_vec = torch.randn(N)
            torch.save(self.teacher_vec, teacherFilename)
        self.N = N
    def make_data(self,P,P_test, trainsetFilename, device):
        resume_status = False 
        try:
            if self.resume: 
                loaded = torch.load(trainsetFilename, map_location=torch.device(device))
                inputs, targets, test_inputs, test_targets = loaded['inputs'], loaded['targets'], loaded['test_inputs'],loaded['test_targets']
                print("\nTrainset was loaded from checkpoint")
                resume_status = True  
            else:
                raise Exception()
        except:
            print("\nCreating new dataset..")
            inputs = torch.randn((P,self.N))
            targets = torch.sum(self.teacher_vec * inputs, dim=1)/self.N
            test_inputs = torch.randn((P_test,self.N))
            test_targets =torch.sum(self.teacher_vec * test_inputs, dim=1)/self.N
        if self.save_data and not resume_status:
            save_data(inputs,targets,test_inputs,test_targets, trainsetFilename)
        return  inputs, targets, test_inputs, test_targets,resume_status

class random_dataset: 
    def __init__(self, N):
        self.N = N
    def make_data(self,P,P_test,  trainsetFilename, device):
        inputs = torch.randn((P,self.N))
        resumed = False
        targets = torch.randn(P)
        test_inputs = torch.randn((P_test,self.N))
        test_targets = torch.randn(P_test)
        return inputs, targets, test_inputs, test_targets,resumed




