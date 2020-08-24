#get rank with respect to each class in a dataset, do this in a hacky single class way, because for some stupid reason your gpu memory is getting used up otherwise
import time
import torch
from subprocess import call
import os
import argparse
from copy import deepcopy
from dissected_Conv2d import *
from torch.autograd import Variable
import sys
sys.path.insert(0, os.path.abspath('../'))

os.chdir('../')
import prep_model_parameters as params
os.chdir('./prep_model_scripts')


#command Line argument parsing
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--orig-data-path", type = str, default = params.rank_img_path)
	parser.add_argument("--dummy-path", type = str, default = None)
	parser.add_argument("--label", type = str, default = None)
	parser.add_argument("--output_folder", type = str, default = params.output_folder)
	parser.add_argument('--cuda', action='store_true', default=params.cuda, help='Use CPU not GPU')  
	parser.add_argument('--seed', type=int, default=params.seed, metavar='S',
						help='random seed (default set in parameters file)')
	parser.add_argument('--batch-size', type=int, default=params.batch_size, metavar='BS',
						help='size of batch to run through model at a time (default set in parameter file)')
	parser.add_argument('--num-workers', type=int, default=params.num_workers, metavar='W',
						help='number of parallel batches to process. Rule of thumb 4*num_gpus (default set in parameters file)')

	args = parser.parse_args()
	return args


start = time.time()

args = get_args()

if args.dummy_path is None or args.label is None:
	raise ValueError('must specify dummy_path and label in function call, can\'t be None')

#populate label folder with links
call('rmdir %s'%os.path.join(args.dummy_path,args.label),shell=True)
call('ln -s %s/ %s'%(os.path.join(args.orig_data_path,args.label),os.path.join(args.dummy_path,args.label)),shell=True)


torch.manual_seed(args.seed)

##MODEL LOADING

model = params.model

if args.cuda:
	model = model.cuda()
else:
	model = model.cpu()


model_dis = dissect_model(deepcopy(model),cuda=params.cuda) #version of model with accessible preadd activations in Conv2d modules 

del model
torch.cuda.empty_cache()


##DATA LOADER###
import torch.utils.data as data
import torchvision.datasets as datasets

image_loader = data.DataLoader(
        			datasets.ImageFolder(args.dummy_path, params.preprocess),
        			batch_size=args.batch_size,
        			shuffle=True,
        			num_workers=args.num_workers,
        			pin_memory=False)


##RUNNING DATA THROUGH MODEL
for param in model_dis.parameters():  #need gradients for grad*activation rank calculation
	param.requires_grad = True

node_ranks = {}
edge_ranks = {}

def order_target(target,order_file):
	file = open(order_file,'r')
	reorder = [x.strip() for x in file.readlines()]
	current_order = deepcopy(reorder)
	current_order.sort()      #current order is alphabetical 
	if len(target.shape)==1:
		for i in range(len(target)):
			class_name = current_order[target[i]]
			target[i] = reorder.index(class_name)
		file.close()
		return target
	elif len(target.shape)==2:
		sys.exit('only 1 dimensional target vectors currently supported, not 2 :(')
	else:
		sys.exit('target has incompatible shape for reordering: %s'%str(target.shape))


#Pass data through model in batches
for i, (batch, target) in enumerate(image_loader):
	print('batch %s'%i)
	model_dis.zero_grad()
	batch = Variable(batch)
	if os.path.exists(os.path.join(params.rank_img_path,'label_order.txt')):
		target = order_target(target,os.path.join(params.rank_img_path,'label_order.txt'))
	if params.cuda:
		batch = batch.cuda()
		target = target.cuda()

	output = model_dis(batch)    #running forward pass sets up hooks and stores activations in each dissected_Conv2d module
	try:
		params.criterion(output, Variable(target)).backward()    #running backward pass calls all the hooks and calculates the ranks of all edges and nodes in the graph 
	except:
		torch.sum(output).backward()    # run backward pass with respect to net outputs rather than loss function

##FETCHING RANKS

def get_ranks_from_dissected_Conv2d_modules(module,prenorm=False,layer_ranks=None):     #run through all model modules recursively, and pull the ranks stored in dissected_Conv2d modules 
	if layer_ranks is None:    #initialize the output dictionary if we are not recursing and havent done so yet
		layer_ranks = {'nodes':{'act':[],'grad':[],'weight':[],'actxgrad':[]},'edges':{'act':[],'weight':[],'grad':[],'actxgrad':[]}}
	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, dissected_Conv2d):
			submodule.average_ranks()
			submodule.normalize_ranks()
			for key in ['act','grad','actxgrad','weight']:
				if not prenorm:
					layer_ranks['nodes'][key].append(submodule.postbias_ranks[key].cpu().detach().numpy())
					layer_ranks['edges'][key].append(submodule.format_edges(data= 'ranks')[key])
				else:
					layer_ranks['nodes'][key].append(submodule.postbias_ranks_prenorm[key].cpu().detach().numpy())
					layer_ranks['edges'][key].append(submodule.format_edges(data= 'ranks',prenorm = True)[key])					
				#print(layer_ranks['edges'][-1].shape)
		elif len(list(submodule.children())) > 0:
			layer_ranks = get_ranks_from_dissected_Conv2d_modules(submodule,layer_ranks=layer_ranks)   #module has modules inside it, so recurse on this module
	return layer_ranks


layer_ranks = get_ranks_from_dissected_Conv2d_modules(model_dis)
layer_ranks_prenorm = get_ranks_from_dissected_Conv2d_modules(model_dis,prenorm=True)




##SAVE LABEL RANKS##
os.makedirs('../prepped_models/'+args.output_folder+'/ranks/',exist_ok=True)
torch.save(layer_ranks, '../prepped_models/'+args.output_folder+'/ranks/%s_rank.pt'%args.label)

os.makedirs('../prepped_models/'+args.output_folder+'/extra_data/',exist_ok=True)
torch.save(layer_ranks_prenorm, '../prepped_models/'+args.output_folder+'/extra_data/%s_prenorm_rank.pt'%args.label)


#remove symlinks from dummy folder
call('rm %s'%os.path.join(args.dummy_path,args.label),shell=True)
os.mkdir(os.path.join(args.dummy_path,args.label))


print('single class rank time: %s'%str(time.time()-start))
