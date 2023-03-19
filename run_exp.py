import utils, teachers 
from utils import CorrMat, kmatrix, kernel_erf,torch
import numpy as np

def run_experiment(realization_id, args,inputs, targets,test_inputs,test_targets,device,run_folder):
    # ... [put the whole code you provided here]
    start_time = utils.time.time()
    #args = utils.parseArguments()
    print(f"\nThe input dimension is {args.N} \nNumber of examples in the train set: {args.P} \nNumber of examples in the test set: {args.Ptest}" )

    bias = False
    net_class = utils.ConvNet(args.K, args.f, args.L, args.lambda1, args.lambda0)
    net, norm = net_class.Sequential(1, int(utils.np.sqrt(args.N)), args.stride, bias, args.act)
    utils.cuda_init(net, device)

    #with torch.no_grad():    
    #    smallnet = net[0:-1]
    #    #print(smallnet)
    #    feature_vec = smallnet(inputs)
    #    Kstart = CorrMat(feature_vec,args.lambda0)
    with torch.no_grad():
        smallnet = net[0:-1]
        post_acts = smallnet(inputs)
        last_layer = net[-1]
        weights = last_layer.weight.flatten()
        features = []
        output_size= len(weights)//args.K
        #print(post_acts)
        #print(weights)
        for mu in range(args.P):
            local_feat = torch.stack(torch.split(post_acts[mu].squeeze(0) * weights , output_size))
            features.append(torch.sum(local_feat,1)) 
        #NTK FEATURE VEC #feature_vec=[torch.dot(post_acts[j],weights) for j in range(args.P)]
        #Kend = CorrMat(feature_vec,args.lambda0)
        Kstart = utils.Conv_ker(features,norm)
    

    #TRAINING DYNAMICS SPECIFICATION
    if args.opt == 'adam': 
    	optimizer = utils.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
    elif args.opt == 'sgd':
    	optimizer = utils.optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
    criterion = utils.nn.MSELoss()
    if args.bs == 0: 
    	batch_size_train,batch_size_test =args.P, min(args.Ptest, args.P)
    else: 
    	batch_size_train, batch_size_test = args.bs, args.bs
    train_args = [net,inputs,targets,criterion,optimizer,batch_size_train,norm]	
    test_args = [net,test_inputs,test_targets,criterion,batch_size_test,norm]	
    training = utils.train
    testing = utils.test

    #RUN DYNAMICS
    runFilename = f'{run_folder}run_P_{args.P}_replica_{realization_id}.txt'
    if not utils.os.path.exists(runFilename):
    	with open(runFilename, 'a') as f:
    		print('#1. epoch', '2. train error', '3. test error', file = f)
    sourceFile = open(runFilename, 'a')
    for epoch in range(args.epochs):
    	train_loss = training(*train_args)
    	#test_loss = testing(*test_args)
    	print(epoch, train_loss, " ", file = sourceFile)
    	if epoch % args.checkpoint == 0 or epoch == args.epochs-1 :
            test_loss = testing(*test_args)
            print(epoch, train_loss, test_loss, file = sourceFile)
            sourceFile.close()
            sourceFile = open(runFilename, 'a')
            print(f'\nEpoch: {epoch} \nTrain error: {train_loss} \nTest error: {test_loss} \n training took --- {utils.time.time() - start_time} seconds ---')				
            start_time = utils.time.time()	
    sourceFile.close()

    with torch.no_grad():
        smallnet = net[0:-1]
        post_acts = smallnet(inputs)
        last_layer = net[-1]
        weights = last_layer.weight #.flatten()
        features = []
        output_size= len(weights)//args.K
        for mu in range(args.P):
            local_feat = torch.stack(torch.split(post_acts[mu] * weights , output_size))
            features.append(torch.sum(local_feat,1)) 
        #NTK FEATURE VEC #feature_vec=[torch.dot(post_acts[j],weights) for j in range(args.P)]
        #Kend = CorrMat(feature_vec,args.lambda0)
        Kend = utils.Conv_ker(features,norm)
        #Kend = 0

    return Kstart, Kend

def main():
    # Parse arguments and set up other configurations here
    args = utils.parseArguments()
    num_realizations = args.R  # Set the number of realizations you want to average over

    #CREATION OF FOLDERS AND TEACHERS
    device = utils.find_device(args.device)
    home = utils.os.environ['HOME']
    #mother_dir = f'{home}/deepL_beyond_infwidth_runs/'
    mother_dir = f'./runs/'
    utils.make_directory(mother_dir)
    #global first_subdir, run_folder, teacher_class
    first_subdir, run_folder = utils.make_folders(mother_dir, args)
    trainsetFilename = "temp"
    if args.teacher_type == 'linear': 
    	teacher_dir = f'{mother_dir}linear_teachers_trainsets/'
    	utils.make_directory(teacher_dir)
    	teacherFilename = f'{teacher_dir}/teacher_N_{args.N}.pt'
    	trainsetFilename = f'{teacher_dir}/trainset_N_{args.N}_P_{args.P}_Ptest_{args.Ptest}.pt'
    	teacher_class = teachers.linear_dataset()
    	teacher_class.resume = args.resume_data
    	teacher_class.make_teacher_parameters(args.N, teacherFilename)
    elif args.teacher_type == 'mnist': 
    	teacher_class = teachers.MNISTDataset(args.N)
    elif args.teacher_type == 'random': 
    	teacher_class = teachers.random_dataset(args.N)
    teacher_class.save_data = args.save_data

    #CRREATION OF DADTASET 
    inputs, targets, test_inputs, test_targets, resumed = teacher_class.make_data(args.P, args.Ptest, trainsetFilename, device)
    inputs, targets, test_inputs, test_targets = inputs.to(device), targets.to(device), test_inputs.to(device), test_targets.to(device)

    Kstart_accum = np.zeros((args.P, args.P))
    Kend_accum = np.zeros((args.P, args.P))
    
    for realization_id in range(num_realizations):
        Kstart, Kend = run_experiment(realization_id, args, inputs,targets,test_inputs,test_targets,device,run_folder)
        Kstart_accum += Kstart
        Kend_accum += Kend

    Kstart_avg = Kstart_accum / num_realizations
    Kend_avg = Kend_accum / num_realizations



    # Save the averaged Kstart and Kend matrices
    kfile_start_avg = f"{run_folder}/conv_start_kernel_avg.txt"
    with open(kfile_start_avg, "a") as f:
        for i in range(args.P):
            for j in range(args.P):
                print(Kstart_avg[i][j], end=" ", file=f)
            print("\n", end="", file=f)

    kfile_end_avg = f"{run_folder}/conv_end_kernel_avg.txt"
    with open(kfile_end_avg, "a") as f:
        for i in range(args.P):
            for j in range(args.P):
                print(Kend_avg[i][j], end=" ", file=f)
            print("\n", end="", file=f)

    kfile_diff_avg = f"{run_folder}/conv_diff_kernel_avg.txt"
    with open(kfile_diff_avg,"a") as f:
        for i in range(args.P):
            for j in range(args.P):
                print(Kend_avg[i][j]/Kstart_avg[i][j], end=" ", file = f)
            print("\n", end = "",file = f)

if __name__ == "__main__":
    main()