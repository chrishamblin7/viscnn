#get rank with respect to each class in a dataset, do this in a hacky single class way, because for some stupid reason your gpu memory is getting used up otherwise

from torchvision import models
from subprocess import call
import os
import sys
sys.path.append('../')
from finetune import Pruning_Finetuner, FilterPrunner
from modified_models import *
import argparse
import time
import pdb

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", type = str, default = "/home/chris/projects/categorical_pruning/data/letters_and_enumeration/train")
	parser.add_argument("--dummy_path", type = str, default = "/home/chris/projects/categorical_pruning/data/letters_and_enumeration/train_dummy")
	parser.add_argument("--test_path", type = str, default = "/home/chris/projects/categorical_pruning/data/letters_and_enumeration/test")
	parser.add_argument("--model", type = str, default = '../models/cifar_prunned_.816')
	parser.add_argument("--label", type = str, default = '1')
	parser.add_argument("--output_folder", type = str, default = "../rankings/cifar")
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Use CPU not GPU')  
	parser.add_argument('--seed', type=int, default=2, metavar='S',
						help='random seed (default: 2)')
	parser.add_argument("--cuda_device", type = str, default = "0")
	parser.add_argument('--multi-gpu', action='store_true', default=False,
						help='run model on multiple gpus')
	parser.add_argument('--num-workers', type=int, default=4, metavar='W',
						help='number of parallel batches to process. Rule of thumb 4*num_gpus (default: 4)')
	#parser.add_argument("--no-crop", dest="no_crop", action="store_true") 
	#parser.add_argument('--grayscale', action='store_true', default=False,
	#					help='load images as grayscale, for models that take grayscale input') 
	parser.add_argument('--indiv-acc', action='store_true', default=False,
						help='output individual class accuracies and f1 scores when testing model')

	parser.set_defaults(no_crop=False)
	args = parser.parse_args()
	#args.use_cuda = args.use_cuda and torch.cuda.is_available()

	return args


args = get_args()

args.start = time.time()

torch.manual_seed(args.seed)

model = torch.load(args.model, map_location=lambda storage, loc: storage)
#model = models.alexnet(pretrained=True)

if  not args.no_cuda:

	model = model.cuda()

if args.multi_gpu:
	os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3"
	#model = _CustomDataParallel(model)
	model= nn.DataParallel(model)

train_path = args.train_path
test_path = args.test_path
args.train_path = args.dummy_path
args.test_path = args.dummy_path



#populate label folder with links
call('rmdir %s'%os.path.join(args.dummy_path,args.label),shell=True)
call('ln -s %s/ %s'%(os.path.join(train_path,args.label),os.path.join(args.dummy_path,args.label)),shell=True)

#pdb.set_trace()
fine_tuner = Pruning_Finetuner(model, args)

#fine_tuner.test()
#pdb.set_trace()
rank = fine_tuner.rank()
torch.save(rank, os.path.join(args.output_folder,"%s_rank.pt"%args.label))


#remove links
call('rm %s'%os.path.join(args.dummy_path,args.label),shell=True)
os.mkdir(os.path.join(args.dummy_path,args.label))

print('time: %s'%str(time.time() - args.start))


