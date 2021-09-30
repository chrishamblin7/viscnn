#get rank with respect to each category in a dataset, do this in a hacky single category way, because for some stupid reason your gpu memory is getting used up otherwise
import time
import torch
from subprocess import call
import os
import argparse
from copy import deepcopy
from dissected_Conv2d import *
from data_loading_functions import *
from torch.autograd import Variable
import pandas as pd
import sys

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("output_folder", type = str, help='the folder name for this prepped model')
	parser.add_argument("--category", type = str, default = None)
	parser.add_argument("--data-path", type = str, default = None)
	#parser.add_argument("--parent-data-path", type = str, default = params.rank_img_path)
	#parser.add_argument("--data-path", type = str, default = None)
	#parser.add_argument("--category", type = str, default = None)
	#parser.add_argument("--output_folder", type = str, default = params.output_folder)
	#parser.add_argument('--cuda', action='store_true', default=params.cuda, help='Use CPU not GPU')  
	#parser.add_argument('--seed', type=int, default=params.seed, metavar='S',
	#					help='random seed (default set in parameters file)')
	#parser.add_argument('--batch-size', type=int, default=params.batch_size, metavar='BS',
	#					help='size of batch to run through model at a time (default set in parameter file)')
	#parser.add_argument('--num-workers', type=int, default=params.num_workers, metavar='W',
	#					help='number of parallel batches to process. Rule of thumb 4*num_gpus (default set in parameters file)')


	args = parser.parse_args()
	return args

start = time.time()

args = get_args()
output_folder = args.output_folder

sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../visualizer_scripts/'))
from visualizer_helper_functions import get_ranks_from_dissected_Conv2d_modules

sys.path.insert(0, os.path.abspath('../prepped_models/%s'%output_folder))
os.chdir(os.path.abspath('../prepped_models/%s'%output_folder))
import prep_model_params_used as params
os.chdir('../../prep_model_scripts')




if args.data_path is None or args.category is None:
	raise ValueError('must specify data_path and category in function call, can\'t be None')

#populate category folder with links
#call('rmdir %s'%os.path.join(args.dummy_path,args.category),shell=True)
#call('ln -s %s/ %s'%(os.path.join(args.orig_data_path,args.category),os.path.join(args.dummy_path,args.category)),shell=True)

torch.manual_seed(params.seed)

device = torch.device("cuda" if params.cuda else "cpu")

##MODEL LOADING

model_dis = dissect_model(deepcopy(params.model),cuda=params.cuda) #version of model with accessible preadd activations in Conv2d modules 
if params.cuda:
	model_dis.cuda()
del params.model

model_dis = model_dis.eval()
for param in model_dis.parameters():  #need gradients for grad*activation rank calculation
	param.requires_grad = True

##DATA LOADER###
import torch.utils.data as data
import torchvision.datasets as datasets

kwargs = {'num_workers': params.num_workers, 'pin_memory': True} if params.cuda else {}

image_loader = torch.utils.data.DataLoader(
			rank_image_data(args.data_path,params.preprocess,params.label_file_path),
			batch_size=params.batch_size,
			shuffle=True,
			**kwargs)			
			


##RUNNING DATA THROUGH MODEL

node_ranks = {}
edge_ranks = {}

#Pass data through model in batches
for i, (batch, target) in enumerate(image_loader):
	print('batch %s'%i)
	model_dis.zero_grad()
	batch, target = batch.to(device), target.to(device)
	output = model_dis(batch)    #running forward pass sets up hooks and stores activations in each dissected_Conv2d module
	target = max_likelihood_for_no_target(target,output) 
	params.criterion(output, Variable(target)).backward()    #running backward pass calls all the hooks and calculates the ranks of all edges and nodes in the graph 
	#except:
	#	torch.sum(output).backward()    # run backward pass with respect to net outputs rather than loss function

##FETCHING RANKS

#import pdb; pdb.set_trace()

layer_ranks = get_ranks_from_dissected_Conv2d_modules(model_dis)
#layer_ranks_prenorm = get_ranks_from_dissected_Conv2d_modules(model_dis,prenorm=True)

##SAVE category RANKS##
os.makedirs('../prepped_models/'+output_folder+'/ranks/categories_edges/',exist_ok=True)
os.makedirs('../prepped_models/'+output_folder+'/ranks/categories_nodes/',exist_ok=True)
torch.save(layer_ranks['nodes'], '../prepped_models/'+output_folder+'/ranks/categories_nodes/%s_nodes_rank.pt'%args.category)
torch.save(layer_ranks['edges'], '../prepped_models/'+output_folder+'/ranks/categories_edges/%s_edges_rank.pt'%args.category)

#os.makedirs('../prepped_models/'+args.output_folder+'/ranks/prenorm/categories_edges/',exist_ok=True)
#os.makedirs('../prepped_models/'+args.output_folder+'/ranks/prenorm/categories_nodes/',exist_ok=True)
#torch.save(layer_ranks_prenorm['nodes'], '../prepped_models/'+args.output_folder+'/ranks/prenorm/categories_nodes/%s_prenorm_nodes_rank.pt'%args.category)
#torch.save(layer_ranks_prenorm['edges'], '../prepped_models/'+args.output_folder+'/ranks/prenorm/categories_edges/%s_prenorm_edges_rank.pt'%args.category)

#CHECK FOR WEIGHT RANK
if not os.path.exists('../prepped_models/'+output_folder+'/ranks/weight_nodes_ranks.csv'):
	print('generating weight rank csvs')

	weight_ranks = get_ranks_from_dissected_Conv2d_modules(model_dis,weight_rank=True)

	#save node csv
	node_num = 0
	weightnode_dflist = []
	for layer in range(len(weight_ranks['nodes']['weight'])):
		layer_name = weight_ranks['nodes']['weight'][layer][0]
		for num_by_layer in range(len(weight_ranks['nodes']['weight'][layer][1])):
			weightnode_dflist.append([node_num,layer_name,layer,num_by_layer,weight_ranks['nodes']['weight'][layer][1][num_by_layer]])
			node_num += 1
	node_column_names = ['node_num','layer_name','layer','node_num_by_layer','weight_rank']
	node_df = pd.DataFrame(weightnode_dflist,columns=node_column_names)
	#save
	node_df.to_csv('../prepped_models/'+output_folder+'/ranks/weight_nodes_ranks.csv',index=False)

	#save edge csv
	edge_num = 0
	weightedge_dflist = []
	for layer in range(len(weight_ranks['edges']['weight'])):
		layer_name = weight_ranks['edges']['weight'][layer][0]
		for out_channel in range(len(weight_ranks['edges']['weight'][layer][1])):
			for in_channel in range(len(weight_ranks['edges']['weight'][layer][1][out_channel])):
				weightedge_dflist.append([edge_num,layer_name,layer,out_channel,in_channel,weight_ranks['edges']['weight'][layer][1][out_channel][in_channel]])
				edge_num += 1
	edge_column_names = ['edge_num','layer_name','layer','out_channel','in_channel','weight_rank']
	edge_df = pd.DataFrame(weightedge_dflist,columns=edge_column_names)
	#save
	edge_df.to_csv('../prepped_models/'+output_folder+'/ranks/weight_edges_ranks.csv',index=False)

	#torch.save(weight_ranks['nodes'], '../prepped_models/'+args.output_folder+'/ranks/weight_nodes_rank.pt')
	#torch.save(weight_ranks['edges'], '../prepped_models/'+args.output_folder+'/ranks/weight_edges_rank.pt')
	#torch.save(layer_ranks_prenorm['nodes'], '../prepped_models/'+args.output_folder+'/ranks/prenorm/weight_prenorm_nodes_rank.pt')
	#torch.save(layer_ranks_prenorm['edges'], '../prepped_models/'+args.output_folder+'/ranks/prenorm/weight_prenorm_edges_rank.pt')


#remove symlinks from dummy folder
#call('rm %s'%os.path.join(args.dummy_path,args.category),shell=True)
#os.mkdir(os.path.join(args.dummy_path,args.category))


print('single category rank time: %s'%str(time.time()-start))
