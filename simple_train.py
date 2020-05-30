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
import argparse
from operator import itemgetter
from modified_models import *

import time
import pdb

class FineTuner:
	def __init__(self,model, args):
	   
		#self.train_data_loader,self.class_dict = dataset.loader(args.train_path, num_workers=args.num_workers,no_crop=args.no_crop,grayscale=args.grayscale)
		#self.test_data_loader = dataset.test_loader(args.test_path, num_workers=args.num_workers,no_crop=args.no_crop,grayscale=args.grayscale)
		self.train_data_loader,self.class_dict = dataset.loader(args.train_path, num_workers=args.num_workers)
		self.test_data_loader = dataset.test_loader(args.test_path, num_workers=args.num_workers)

		self.model = model
		self.criterion = torch.nn.CrossEntropyLoss()
		self.model.train()
		self.num_classes = args.num_classes
		self.seed = args.seed

	def test(self):
		'''
		self.model.eval()
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
		if acc > .3:
			torch.save(model, "models/model_trained_enumeration_%s"%acc)
		self.model.train()

		'''

		self.model.eval()
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

		if correct / len(self.test_data_loader.dataset) > .81:
			torch.save(self.model, "models/cifar_prunned_%s"%(str(round(acc,3))))

		print('class    total     guessed    accuracy    f1-score')

		for prednum in indiv_acc_dict:
			if indiv_acc_dict[prednum][0] == 0:
				print('no samples for class %s'%str(prednum+1))
			else:
				numclass = prednum
				total = indiv_acc_dict[prednum][0]
				guessed = indiv_acc_dict[prednum][2]
				accuracy = round(indiv_acc_dict[prednum][1]/indiv_acc_dict[prednum][0],3)
				f1 = round(total*accuracy/(total+guessed)*2,3)
				print('%s        %s         %s        %s         %s'%(str(numclass),str(total),str(guessed),str(accuracy),str(f1)))
				indiv_acc_dict[prednum].append(','.join([str(round(accuracy,3)),str(guessed),str(round(f1,3))]))






	def train(self, optimizer = None, epoches=10):
		if optimizer is None:
			#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8, nesterov=True)
			optimizer = optim.Adam(model.parameters(), lr=0.001)
		self.test()
		for i in range(epoches):
			print("Epoch: ", i)
			self.train_epoch(optimizer)
			self.test()
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
		else:
			self.criterion(self.model(input), Variable(label)).backward()
			optimizer.step()

	def train_epoch(self, optimizer = None, rank_filters = False):
		for i, (batch, label) in enumerate(self.train_data_loader):
			#if i == 0:
			#	torch.save(Variable(batch),'lettersandenum_batch.pt')
			#	exit()
			if i%10==0:
				print('batch '+str(i) + ' time: ' + str(time.time() - start))
			#print('batch shape ' + str(batch.shape))
			self.train_batch(optimizer, batch, label, rank_filters)

	def get_class(self,val): 
		for key, value in self.class_dict.items(): 
			if val == value: 
				return key 


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train", dest="train", action="store_true")
	parser.add_argument("--test", dest="test", action="store_true")
	parser.add_argument("--train-path", type = str, default = "../data/cifar10/train")
	parser.add_argument("--test-path", type = str, default = "../data/cifar10/test")
	parser.add_argument("--load-model", type = str, default = "models/cifar_small_0.821_prunned_prunned")
	parser.add_argument('--use-cuda', action='store_true', default=False, help='Use NVIDIA GPU acceleration')  
	parser.add_argument('--seed', type=int, default=2, metavar='S',
						help='random seed (default: 2)')
	parser.add_argument('--num-classes', type=int, default=10, metavar='S',
						help='number of output classes (default: 21)')
	parser.add_argument("--cuda_device", type = str, default = "0")
	parser.add_argument('--multi-gpu', action='store_true', default=False,
						help='run model on multiple gpus')
	parser.add_argument('--num-workers', type=int, default=4, metavar='W',
						help='number of parallel batches to process. Rule of thumb 4*num_gpus (default: 4)')
	#parser.add_argument("--no-crop", dest="no_crop", action="store_true") 
	#parser.add_argument('--grayscale', action='store_true', default=False,
	#					help='load images as grayscale, for models that take grayscale input')  
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

	start = time.time()

	torch.manual_seed(args.seed)

	if args.train:

		#model = cifar_CNN()

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

	fine_tuner = FineTuner(model,args)
	#fine_tuner = FineTuner(args.train_path, args.test_path, model, args.num_workers, args.no_crop,args.num_classes)

	if args.train:
		
		fine_tuner.train(epoches=100)
		#torch.save(model, "model_enumeration_49adam_lr005")


	elif args.test:
		fine_tuner.test()


else:
	args = DefaultArgs()
	import importlib




'''

nb_classes = 9

confusion_matrix = torch.zeros(nb_classes, nb_classes)
with torch.no_grad():
	for i, (inputs, classes) in enumerate(dataloaders['val']):
		inputs = inputs.to(device)
		classes = classes.to(device)
		outputs = model_ft(inputs)
		_, preds = torch.max(outputs, 1)
		for t, p in zip(classes.view(-1), preds.view(-1)):
				confusion_matrix[t.long(), p.long()] += 1

print(confusion_matrix)
#To get the per-class accuracy:

print(confusion_matrix.diag()/confusion_matrix.sum(1))
'''