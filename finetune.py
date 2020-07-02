import torch
from torch.autograd import Variable
from torchvision import models
import sys
import os
import numpy as np
import cv2
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import dataset
from prune import *
import argparse
from operator import itemgetter
from heapq import nsmallest
import time
import pdb
from modified_models import *


class FilterPrunner:
	def __init__(self, model):
		self.model = model
		self.reset()
	
	def reset(self):
		self.filter_ranks = {}

	def forward(self, x):
		self.activations = []
		self.gradients = []
		self.grad_index = 0
		self.activation_to_layer = {}

		activation_index = 0
		for layer, (name, module) in enumerate(self.model.features._modules.items()):
			x = module(x)
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				x.register_hook(self.compute_rank)
				self.activations.append(x)
				self.activation_to_layer[activation_index] = layer
				activation_index += 1

		#### EDITED FOR CUSTOM RESNET BASED MODEL        
		x = self.model.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.model.classifier(x)
		return x
		#return self.model.classifier(x.view(x.size(0), -1))

	def compute_rank(self, grad):
		#pdb.set_trace()
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]

		taylor = activation * grad
		# Get the average value for every filter, 
		# accross all the other dimensions
		taylor = taylor.mean(dim=(0, 2, 3)).data


		if activation_index not in self.filter_ranks:
			self.filter_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_()

			if args.use_cuda:
				self.filter_ranks[activation_index] = self.filter_ranks[activation_index].cuda()

		self.filter_ranks[activation_index] += taylor
		self.grad_index += 1

	def lowest_ranking_filters(self, num):
		data = []
		#
		for i in sorted(self.filter_ranks.keys()):
			for j in range(self.filter_ranks[i].size(0)):
				data.append((self.activation_to_layer[i], j, self.filter_ranks[i][j]))

		return nsmallest(num, data, itemgetter(2))

	def normalize_ranks_per_layer(self):
		for i in self.filter_ranks:
			v = torch.abs(self.filter_ranks[i])
			v = v.cpu()
			v = v / np.sqrt(torch.sum(v * v))
			self.filter_ranks[i] = v

	def get_prunning_plan(self, num_filters_to_prune):
		filters_to_prune = self.lowest_ranking_filters(num_filters_to_prune)

		# After each of the k filters are prunned,
		# the filter index of the next filters change since the model is smaller.
		filters_to_prune_per_layer = {}
		for (l, f, _) in filters_to_prune:
			if l not in filters_to_prune_per_layer:
				filters_to_prune_per_layer[l] = []
			filters_to_prune_per_layer[l].append(f)

		for l in filters_to_prune_per_layer:
			filters_to_prune_per_layer[l] = sorted(filters_to_prune_per_layer[l])
			for i in range(len(filters_to_prune_per_layer[l])):
				filters_to_prune_per_layer[l][i] = filters_to_prune_per_layer[l][i] - i

		filters_to_prune = []
		for l in filters_to_prune_per_layer:
			for i in filters_to_prune_per_layer[l]:
				filters_to_prune.append((l, i))

		return filters_to_prune             

class Pruning_Finetuner:
	def __init__(self, model, args):
	   
		#self.train_data_loader,self.class_dict = dataset.loader(args.train_path, num_workers=args.num_workers,no_crop=args.no_crop,grayscale = args.grayscale)
		#self.test_data_loader = dataset.test_loader(args.test_path, num_workers=args.num_workers,no_crop=args.no_crop,grayscale = args.grayscale)
		self.train_data_loader,self.class_dict = dataset.loader(args.train_path, num_workers=args.num_workers)
		self.test_data_loader = dataset.test_loader(args.test_path, num_workers=args.num_workers)

		self.model = model
		self.num_classes = self.model.out_channels
		self.criterion = torch.nn.CrossEntropyLoss()
		self.prunner = FilterPrunner(self.model) 
		
		self.start = args.start
		self.indiv_acc = args.indiv_acc
		self.model.train()
		self.args = args

		

	def test(self):
		self.model.eval()

		if self.indiv_acc:
			test_loss = 0
			correct = 0
			indiv_acc_dict = {}
			for i in range(self.num_classes):
				indiv_acc_dict[self.get_class(i)] = [0,0,0]
			with torch.no_grad():
				for data,target in self.test_data_loader:
					if args.use_cuda:
						data, target = data.cuda(), target.cuda()
					output = self.model(data)
					test_loss += self.criterion(output, target).item()
					pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
					
					correct += pred.eq(target.view_as(pred)).sum().item()

				   
					for i in range(len(pred)):
						indiv_acc_dict[self.get_class(int(pred[i]))][2] += 1          
						indiv_acc_dict[self.get_class(int(target.view_as(pred)[i]))][0] += 1
						if int(pred[i]) == int(target.view_as(pred)[i]):
							indiv_acc_dict[self.get_class(int(pred[i]))][1] += 1
							
			test_loss /= len(self.test_data_loader.dataset)

			acc = correct / len(self.test_data_loader.dataset)

			print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
				test_loss, correct, len(self.test_data_loader.dataset),
				100. * correct / len(self.test_data_loader.dataset)))

			#if correct / len(self.test_data_loader.dataset) > .9:
			#	torch.save(self.model, "models/custom_trained_enumandletters_%s"%acc)

			print('class    total     guessed    accuracy    f1-score')
			for label in indiv_acc_dict:
				if indiv_acc_dict[label][0] == 0:
					print('no samples for class %s'%str(label))
				else:
					#numclass = label
					total = indiv_acc_dict[label][0]
					guessed = indiv_acc_dict[label][2]
					accuracy = round(indiv_acc_dict[label][1]/indiv_acc_dict[label][0],3)
					f1 = round(total*accuracy/(total+guessed)*2,3)
					print('%s        %s         %s        %s         %s'%(str(label),str(total),str(guessed),str(accuracy),str(f1)))
					indiv_acc_dict[label].append(','.join([str(round(accuracy,3)),str(guessed),str(round(f1,3))]))
		else:
			correct = 0
			total = 0
			for i, (batch, label) in enumerate(self.test_data_loader):
				if args.use_cuda:
					batch = batch.cuda()
				#print('batch shape ' + str(batch.shape))
				output = self.model(Variable(batch))
				pred = output.data.max(1)[1]
				correct += pred.cpu().eq(label).sum()
				total += label.size(0)
			acc = float(correct) / total
			print("Accuracy :", acc)
			#if acc > .3:
			#    torch.save(model, "models/model_enumeration_%s"%acc)
		self.model.train()
		return acc

	def train(self, optimizer = None, max_epoches = 10, min_acc=.95):
		if optimizer is None:
			#optimizer = optim.SGD(model.classifier.parameters(), lr=0.0001, momentum=0.9)
			optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
		acc = self.test()
		for i in range(max_epoches):
			if acc > min_acc:
				break
			print("Epoch: ", i)
			self.train_epoch(optimizer)
			acc = self.test()
			
		print("Finished fine tuning.")
		

	def train_batch(self, optimizer, batch, label, rank_filters):

		if args.use_cuda:
			batch = batch.cuda()
			label = label.cuda()

		self.model.zero_grad()
		input = Variable(batch)

		if rank_filters:
			output = self.prunner.forward(input)
			
			self.criterion(output, Variable(label)).backward()
			#pdb.set_trace()
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		#pdb.set_trace()
		for i, (batch, label) in enumerate(self.train_data_loader):

			if i%10==0:
				print('batch '+str(i) + ' time: ' + str(time.time() - self.start))
			#print('batch shape ' + str(batch.shape))
			#pdb.set_trace()
			self.train_batch(optimizer, batch, label, rank_filters)
	
	def get_candidates_to_prune(self, num_filters_to_prune):
		self.prunner.reset()
		self.train_epoch(rank_filters = True)
		
		self.prunner.normalize_ranks_per_layer()
		return self.prunner.get_prunning_plan(num_filters_to_prune)
		
	def total_num_filters(self):
		filters = 0
		for name, module in self.model.features._modules.items():
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				filters = filters + module.out_channels
		return filters

	def rank(self):     # function just for returning the ranking of filters but not actually pruning
		for param in self.model.features.parameters():
			param.requires_grad = True
		self.prunner.reset()
		self.train_epoch(rank_filters = True)
		self.prunner.normalize_ranks_per_layer()
		formatted_rank = []
		unique_id = 0
		for i in sorted(self.prunner.filter_ranks.keys()):
			for j in range(self.prunner.filter_ranks[i].size(0)):
				formatted_rank.append((unique_id,i, j, float(self.prunner.filter_ranks[i][j])))
				unique_id += 1
		return formatted_rank

	def prune(self):
		#Get the accuracy before prunning
		#print(self.model)
		self.test()
		self.model.train()

		#Make sure all the layers are trainable
		for param in self.model.features.parameters():
			param.requires_grad = True

		starting_number_of_filters = self.total_num_filters()
		num_filters_to_prune_per_iteration = self.args.prune_rate

		#iterations = int(float(starting_number_of_filters) / num_filters_to_prune_per_iteration)

		#iterations = int(iterations * 2.0 / 3)

		iterations = self.args.prune_iter
		print("Number of prunning iterations to reduce 67% filters: ", iterations)

		for _ in range(iterations):
			
			print("Ranking filters.. ")
			prune_targets = self.get_candidates_to_prune(num_filters_to_prune_per_iteration)

			layers_prunned = {}
			for layer_index, filter_index in prune_targets:
				if layer_index not in layers_prunned:
					layers_prunned[layer_index] = 0
				layers_prunned[layer_index] = layers_prunned[layer_index] + 1 

			print("Layers that will be prunned", layers_prunned)
			print("Prunning filters.. ")
			model = self.model.cpu()
			for layer_index, filter_index in prune_targets:
				model = prune_conv_layer(model, layer_index, filter_index, use_cuda=self.args.use_cuda)

			self.model = model
			if args.use_cuda:
				self.model = self.model.cuda()

			message = str(100*float(self.total_num_filters()) / starting_number_of_filters) + "%"
			print("Filters remaining", str(message))
			#pdb.set_trace()
			print("Fine tuning to recover from prunning iteration.")
			optimizer = optim.SGD(self.model.parameters(), lr=0.0001, momentum=0.9)
			self.train(max_epoches = 10, min_acc = self.args.acc_thresh)


		print("Finished. Going to fine tune the model a bit more")
		self.train(optimizer, max_epoches=15)
		#torch.save(model, "%s_prunned"%self.args.load_model.replace('models/','models/prunned/'))
		torch.save(model, "%s_prunned"%self.args.load_model)

	def get_class(self,val): 
		for key, value in self.class_dict.items(): 
			if val == value: 
				return key 

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", dest="train", action="store_true")
	parser.add_argument("--test", dest="test", action="store_true")
	parser.add_argument("--prune", dest="prune", action="store_true")
	parser.add_argument("--rank", dest="rank", action="store_true")
	parser.add_argument("--train_path", type = str, default = "../data/cifar10/train")
	parser.add_argument("--test-path", type = str, default = "../data/cifar10/test")
	parser.add_argument("--load-model", type = str, default = "models/cifar_prunned_.816")
	parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')  
	parser.add_argument('--seed', type=int, default=2, metavar='S',
						help='random seed (default: 2)')
	parser.add_argument('--prune-rate', type=int, default=100, metavar='W',
						help='When pruning, how many filters should be deleted each iteration (default: 100)')
	parser.add_argument('--prune-iter', type=int, default=4, metavar='W',
						help='how many pruning iterations to perform (default: 4)')
	parser.add_argument('--acc-thresh', type=float, default=.97, metavar='W',
						help='When pruning, how many filters should be deleted each iteration (default: .97)')	
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
	parser.set_defaults(train=False)
	parser.set_defaults(test=False)
	parser.set_defaults(prune=False)
	parser.set_defaults(rank=False)
	parser.set_defaults(no_crop=False)
	args = parser.parse_args()
	args.use_cuda = args.use_cuda and torch.cuda.is_available()

	return args

class DefaultArgs:
	def __init__(self):
		self.train = False
		self.prune = True
		self.train_path = "../data/letters/data/train"
		self.test_path = "../data/letters/data/test"
		self.use_cuda = True


if __name__ == '__main__':
	args = get_args()

	args.start = time.time()

	torch.manual_seed(args.seed)



	if args.train:
		#model = ModifiedAlexnetModel(out_channels=21)
		model = torch.load(args.load_model, map_location=lambda storage, loc: storage)
	else:
		#
		model = torch.load(args.load_model, map_location=lambda storage, loc: storage)

	if args.use_cuda:

		model = model.cuda()

	if args.multi_gpu:
		os.environ["CUDA_VISIBLE_DEVICES"]="0,2,3"
		#model = _CustomDataParallel(model)
		model= nn.DataParallel(model)




	fine_tuner = Pruning_Finetuner(model, args)

	if args.train:
		
		fine_tuner.train(epoches=100)
		#torch.save(model, "model_enumeration_49adam_lr005")

	elif args.prune:
		print(model)
		fine_tuner.prune()

	elif args.rank:
		rank = fine_tuner.rank()
		#torch.save(rank, "rankings/%s_rank.pt"%args.load_model.split('/')[-1])
		torch.save(rank, "rankings/cifar/overall_rank.pt")

	elif args.test:
		fine_tuner.test()


else:
	args = DefaultArgs()
	import importlib
	#model = torch.load("model", map_location=lambda storage, loc: storage)
	#fine_tuner = Pruning_Finetuner(args.train_path, args.test_path, model)


#run with something like. nohup python finetune.py --train --use-cuda > training_logs/inanimate_6.out>&1 & echo $! > pids/inanimtate_6_pid.txt




