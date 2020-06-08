#get rank with respect to each class in a dataset, do this in a hacky single class way, because for some stupid reason your gpu memory is getting used up otherwise

from torchvision import models
from subprocess import call
import os
import sys
#sys.path.append('../')
from model_classes import *
import argparse
import time
import parameters as params
from ranker import ranker



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

args = get_args()

if args.dummy_path == None or args.label == None:
	raise ValueError('must specify dummy_path and label in function call, can\'t be None')

#populate label folder with links
call('rmdir %s'%os.path.join(args.dummy_path,args.label),shell=True)
call('ln -s %s/ %s'%(os.path.join(args.orig_data_path,args.label),os.path.join(args.dummy_path,args.label)),shell=True)


torch.manual_seed(args.seed)

#model loading
model = params.model

if args.cuda:
	model = model.cuda()
else:
	model = model.cpu()

for param in model.parameters():  #need gradients for grad*activation rank calculation
	param.requires_grad = True


##DATA LOADER###
import torch.utils.data as data
import torchvision.datasets as datasets

image_loader = data.DataLoader(
        			datasets.ImageFolder(args.dummy_path, params.preprocess),
        			batch_size=args.batch_size,
        			shuffle=True,
        			num_workers=args.num_workers,
        			pin_memory=True)


ranker = ranker(model,image_loader,params.criterion,args.cuda)

#Get Node ranks
print('getting node ranks')

node_ranks = ranker.gen_node_ranks()
os.makedirs('../prepped_models/'+args.output_folder+'/ranks/',exist_ok=True)
torch.save(node_ranks, '../prepped_models/'+args.output_folder+'/ranks/%s_rank.pt'%args.label)

#Get Edge ranks
print('getting edge ranks')



#remove links
call('rm %s'%os.path.join(args.dummy_path,args.label),shell=True)
os.mkdir(os.path.join(args.dummy_path,args.label))

