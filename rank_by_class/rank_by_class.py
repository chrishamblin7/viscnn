#get rank with respect to each class in a dataset:

from subprocess import call
import os
import sys
sys.path.append('../')
from finetune import Pruning_Finetuner, FilterPrunner
from modified_models import *
from torchvision import models
import argparse
import time
import pdb

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_path", type = str, default = "/home/chris/projects/categorical_pruning/data/letters_and_enumeration/train")
	parser.add_argument("--test_path", type = str, default = "/home/chris/projects/categorical_pruning/data/letters_and_enumeration/test")
	parser.add_argument("--model", type = str, default = '../models/custom_trained_enumandletters_0.97')
	parser.add_argument("--output_folder", type = str, default = "../rankings/lettersandenum_by_class")
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Use CPU not GPU')  
	parser.add_argument('--seed', type=int, default=2, metavar='S',
						help='random seed (default: 2)')
	parser.add_argument("--cuda_device", type = str, default = "0")
	parser.add_argument('--multi-gpu', action='store_true', default=False,
						help='run model on multiple gpus')
	parser.add_argument('--num-workers', type=int, default=4, metavar='W',
						help='number of parallel batches to process. Rule of thumb 4*num_gpus (default: 4)')
	parser.add_argument("--no-crop", dest="no_crop", action="store_true")  
	parser.add_argument('--indiv-acc', action='store_true', default=False,
						help='output individual class accuracies and f1 scores when testing model')

	parser.set_defaults(no_crop=False)
	args = parser.parse_args()
	#args.use_cuda = args.use_cuda and torch.cuda.is_available()

	return args



def make_dummy_dir(orig_path):
	path_split = orig_path.split('/')
	path_split[-1] = path_split[-1] + '_dummy'
	dummy_path = '/'.join(path_split)
	os.mkdir(dummy_path)
	subdirs = os.listdir(orig_path)
	for subdir in subdirs:
		os.mkdir(os.path.join(dummy_path,subdir))
	return dummy_path, subdirs


if __name__ == '__main__':
	args = get_args()

	args.start = time.time()

	torch.manual_seed(args.seed)

	#model = torch.load(args.model, map_location=lambda storage, loc: storage)
	model = models.alexnet(pretrained=True)

	if  not args.no_cuda:

		model = model.cuda()

	if args.multi_gpu:
		os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3"
		#model = _CustomDataParallel(model)
		model= nn.DataParallel(model)

	if not os.path.exists(args.output_folder):
		os.mkdir(args.output_folder)

	dummy_path, labels = make_dummy_dir(args.train_path)

	train_path = args.train_path
	test_path = args.test_path
	args.train_path = dummy_path
	args.test_path = dummy_path

	for label in labels:
		print(label)
		#populate label folder with links
		call('rmdir %s'%os.path.join(dummy_path,label),shell=True)
		call('ln -s %s/ %s'%(os.path.join(train_path,label),os.path.join(dummy_path,label)),shell=True)

		#pdb.set_trace()
		fine_tuner = Pruning_Finetuner(model, args)

		fine_tuner.test()
		#pdb.set_trace()
		#rank = fine_tuner.rank()
		#torch.save(rank, os.path.join(args.output_folder,"%s_rank.pt"%label))

		#del fine_tuner
		#torch.cuda.empty_cache() 
		#pdb.set_trace()


		#remove links
		call('rm %s'%os.path.join(dummy_path,label),shell=True)
		os.mkdir(os.path.join(dummy_path,label))

	call('rm -r %s'%dummy_path,shell=True)
	print('time: %s'%str(time.time() - start))



#run with something like. nohup python finetune.py --train --use-cuda > training_logs/inanimate_6.out>&1 & echo $! > pids/inanimtate_6_pid.txt
