import utils, teachers 
from theory import compute_theory
from theory import CorrMat, kmatrix, kernel_erf

start_time = utils.time.time()
args = utils.parseArguments()
print(f"\nThe input dimension is {args.N} \nNumber of examples in the train set: {args.P} \nNumber of examples in the test set: {args.Ptest}" )
if args.L>1 and args.act == "erf":
	print("\nI cannot compute predicted gen error of an erf multilayer network")
	args.compute_theory = False 
device = utils.find_device(args.device)
home = utils.os.environ['HOME']

#CREATION OF FOLDERS AND TEACHERS
mother_dir = f'{home}/deepL_beyond_infwidth_runs/'
utils.make_directory(mother_dir)
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

#DEFINE AND INITIALISE NET
bias = False
net_class = utils.ConvNet(args.K, args.f, args.L, args.lambda1, args.lambda0)
net = net_class.Sequential(1, int(utils.np.sqrt(args.N)), args.stride, bias, args.act)
utils.cuda_init(net, device)

#CRREATION OF DADTASET 
inputs, targets, test_inputs, test_targets, resumed = teacher_class.make_data(args.P, args.Ptest, trainsetFilename, device)
inputs, targets, test_inputs, test_targets = inputs.to(device), targets.to(device), test_inputs.to(device), test_targets.to(device)

	

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
train_args = [net,inputs,targets,criterion,optimizer,batch_size_train]	
test_args = [net,test_inputs,test_targets,criterion,batch_size_test]	
training = utils.train
testing = utils.test

smallnet = net[0:3]
print(smallnet)
feature_vec = net(test_inputs)
print(len(feature_vec))
#assert len(feature_vec)==features
Kstart = CorrMat(feature_vec,args.lambda0)
kfile = "conv_start_kernel.txt"
with open(kfile,"a") as f:
    for i in range(args.Ptest):
        for j in range(args.Ptest):
            print(Kstart[i][j], end=" ", file = f)
        print("\n", end = "",file = f)


#RUN DYNAMICS
runFilename = f'{run_folder}run_P_{args.P}_replica_{args.R}.txt'
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

with utils.torch.no_grad():
	
	acts = net[-1]
	print(acts)
	features = len(acts.weight.flatten())
	print(f"\n features in the last layer: {features}")
	net = net[0:3]
	print(net)
	feature_vec = net(test_inputs)
	print(len(feature_vec))
	#assert len(feature_vec)==features
	Kend = CorrMat(feature_vec,args.lambda0)
	kfile = "conv_end_kernel.txt"
	with open(kfile,"a") as f:
	    for i in range(args.Ptest):
	        for j in range(args.Ptest):
	            print(Kend[i][j], end=" ", file = f)
	        print("\n", end = "",file = f)

	kfilediff = "conv_diff.txt"
	with open(kfilediff,"a") as f:
	    for i in range(args.Ptest):
	        for j in range(args.Ptest):
	            print(Kend[i][j]/K[i][j], end=" ", file = f)
	        print("\n", end = "",file = f)