import utils, teachers 
from theory import compute_theory

start_time = utils.time.time()
args = utils.parseArguments()
print(f"\nThe input dimension is {args.N} \nNumber of examples in the train set: {args.P} \nNumber of examples in the test set: {args.Ptest}" )
if args.L>1 and args.act == "erf":
	print("\nI cannot compute predicted gen error of an erf multilayer network")
	args.compute_theory = False 
device = utils.find_device(args.device)
home = utils.os.environ['HOME']

#CREATION OF FOLDERS AND TEACHERS
mother_dir = './runs/prova/'
utils.make_directory(mother_dir)
first_subdir, run_folder = utils.make_folders(mother_dir, args)
trainsetFilename = "temp"
if args.teacher_type == 'linear': 
	teacher_dir = f'{mother_dir}linear_teachers_trainsets/'
	utils.make_directory(teacher_dir)
	teacherFilename = f'{teacher_dir}/teacher_N_{args.N}.pt'
	trainsetFilename = f'{teacher_dir}trainset_N_{args.N}_P_{args.P}_Ptest_{args.Ptest}.pt'
	teacher_class = teachers.linear_dataset()
	teacher_class.resume = args.resume_data
	teacher_class.make_teacher_parameters(args.N, teacherFilename)
elif args.teacher_type == 'mnist': 
	teacher_class = teachers.mnist_dataset(args.N)
elif args.teacher_type == 'random': 
	teacher_class = teachers.random_dataset(args.N)
teacher_class.save_data = args.save_data

#CRREATION OF DADTASET AND THEORY CALCULATION
inputs, targets, test_inputs, test_targets, resumed = teacher_class.make_data(args.P, args.Ptest, trainsetFilename, device)
theoryFilename = f"{first_subdir}theory_N_{args.N}_lambda0_{args.lambda0}_lambda1_{args.lambda1}.txt"
if not utils.os.path.isfile(theoryFilename):
	with open(theoryFilename,"a") as f:
		print('#1P', '2 N1', '3 Qbar', '4 pred error', file = f)
if args.compute_theory:
	start_time = utils.time.time()
	gen_error_pred, Qbar = compute_theory(inputs, targets, test_inputs, test_targets, args.N1, args.lambda1, args.lambda0,args.act,args.L,args.infwidth)
	with open(theoryFilename, "a") as f:
		print(args.P,args.N1,Qbar, gen_error_pred, file = f)
	print(f"\nPredicted error is: {gen_error_pred} \n theory computation took - {utils.time.time() - start_time} seconds -")
	start_time = utils.time.time()
inputs, targets, test_inputs, test_targets = inputs.to(device), targets.to(device), test_inputs.to(device), test_targets.to(device)

#NET INITIALISATION
net_class = utils.make_MLP(args.N, args.N1, args.L, args.lambda1, args.lambda0)
bias = False
net = net_class.Sequential(bias, args.act)
utils.cuda_init(net, device)	

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

#RUN DYNAMICS
runFilename = f'{run_folder}run_P_{args.P}_replica_{args.R}.txt'
if not utils.os.path.exists(runFilename):
	with open(runFilename, 'a') as f:
		print('#1. epoch', '2. train error', '3. test error', file = f)
sourceFile = open(runFilename, 'a')
for epoch in range(args.epochs):
	train_loss = training(*train_args)
	test_loss = testing(*test_args)
	print(epoch, train_loss, test_loss, file = sourceFile)
	if epoch % args.checkpoint == 0 or epoch == args.epochs-1 :
		sourceFile.close()
		sourceFile = open(runFilename, 'a')
		print(f'\nEpoch: {epoch} \nTrain error: {train_loss} \nTest error: {test_loss} \n training took --- {utils.time.time() - start_time} seconds ---')				
		start_time = utils.time.time()	
sourceFile.close()