#Class for ranking importance of edges and nodes in network graph
import torch
from torch.autograd import Variable
from torchvision import models
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dissected_Conv2d import *
from copy import deepcopy

class ranker:
	def __init__(self, model, image_loader, criterion, cuda):
		self.model = model
		self.image_loader = image_loader
		self.criterion = criterion
		self.cuda = cuda
		self.node_ranks = {}
		self.edge_ranks = {}

	def gen_node_ranks(self):
		self.model.zero_grad()
		self.node_ranks = {}

		#Pass data through model in batches
		for i, (batch, target) in enumerate(self.image_loader):
			self.activations = []
			self.grad_index = 0
			self.activation_index = 0

			self.model.zero_grad()
			batch = Variable(batch)
			if self.cuda:
				batch = batch.cuda()
				target = target.cuda()

			x = batch

			###NOTE, THE CODE HERE IS MODEL SPECIFIC, COMMENTED BELOW ARE SOME POSSIBLE GENERAL SOLUTIONS, THAT DONT WORK YET
			for layer, (name, module) in enumerate(self.model.features._modules.items()):
				x = module(x)
				if isinstance(module, torch.nn.modules.conv.Conv2d):
					x.register_hook(self.compute_rank)
					self.activations.append(x)
					self.activation_index += 1

			x = self.model.avgpool(x)
			x = torch.flatten(x, 1)
			output = self.model.classifier(x)

			####SOME POSSIBLE GENERAL SOLUTIONS
			#set up hooks for rank caculation on conv_2d modules
			#for layer, (name, module) in enumerate(self.model._modules.items()):
			#	try:
			#		print(module)
			#		x = module(x)
			#		if isinstance(module, torch.nn.modules.conv.Conv2d):
			#			print('found conv2d')
			#			x.register_hook(self.compute_rank)
			#			self.activations.append(x)
			#			self.activation_index += 1
			#	except:
			#		print('cant continue without specifying forward model')
			#		break

			#alternatively try this recursive thing
			#x = self.set_conv_hooks(x,self.model)

			#run model forward
			#output = self.model(batch)

			######


			
			#run model backward with respect to loss triggering compute rank calculation on the hook
			self.criterion(output, Variable(target)).backward()

		self.normalize_node_ranks_per_layer()
		self.formatted_node_ranks = self.format_node_ranks()
		return self.formatted_node_ranks

	def gen_edge_rank(self):
		self.dis_model = dissect_model(deepcopy(self.model),store_activations=True, store_ranks=True)
		self.dis_model.zero_grad()
		self.edge_ranks = {}

		#Pass data through model in batches
		for i, (batch, target) in enumerate(self.image_loader):
			self.activations = []
			self.grad_index = 0
			self.activation_index = 0

			self.model.zero_grad()
			batch = Variable(batch)
			if self.cuda:
				batch = batch.cuda()
				target = target.cuda()

			x = batch

			###NOTE, THE CODE HERE IS MODEL SPECIFIC, COMMENTED BELOW ARE SOME POSSIBLE GENERAL SOLUTIONS, THAT DONT WORK YET
			for layer, (name, module) in enumerate(self.model.features._modules.items()):
				x = module(x)
				if isinstance(module, torch.nn.modules.conv.Conv2d):
					x.register_hook(self.compute_rank)
					self.activations.append(x)
					self.activation_index += 1

			x = self.model.avgpool(x)
			x = torch.flatten(x, 1)
			output = self.model.classifier(x)

			####SOME POSSIBLE GENERAL SOLUTIONS
			#set up hooks for rank caculation on conv_2d modules
			#for layer, (name, module) in enumerate(self.model._modules.items()):
			#	try:
			#		print(module)
			#		x = module(x)
			#		if isinstance(module, torch.nn.modules.conv.Conv2d):
			#			print('found conv2d')
			#			x.register_hook(self.compute_rank)
			#			self.activations.append(x)
			#			self.activation_index += 1
			#	except:
			#		print('cant continue without specifying forward model')
			#		break

			#alternatively try this recursive thing
			#x = self.set_conv_hooks(x,self.model)

			#run model forward
			#output = self.model(batch)

			######


			
			#run model backward with respect to loss triggering compute rank calculation on the hook
			self.criterion(output, Variable(target)).backward()

		self.normalize_node_ranks_per_layer()
		self.formatted_node_ranks = self.format_node_ranks()
		return self.formatted_node_ranks



	#function passes input through all submodules of model recursively setting hooks on conv2d module outputs
	'''
	def set_conv_hooks(self,x,module):
		for layer, (name, submodule) in enumerate(module._modules.items()):
			if len(list(submodule.children())) > 0:
				x = self.set_conv_hooks(x,submodule)
			else:
				try:
					#print(submodule)
					x = submodule(x)
					if isinstance(submodule, torch.nn.modules.conv.Conv2d):
						#print('found conv2d')
						x.register_hook(self.compute_rank)
						self.activations.append(x)
						self.activation_index += 1
				except:
					#print('cant continue without specifying forward model')
					break
		return x
	'''
		


	def compute_rank(self, grad):
		#print('compute_rank called')
		activation_index = len(self.activations) - self.grad_index - 1
		activation = self.activations[activation_index]

		taylor = activation * grad     #taylor pruning criterion from nvidia paper
		# Get the average value for every activation map, 
		# accross all the other dimensions
		taylor = taylor.mean(dim=(0, 2, 3)).data


		if activation_index not in self.node_ranks:  #if this layer doesnt have any batch values for rank yet, initialize it at all zeros
			self.node_ranks[activation_index] = \
				torch.FloatTensor(activation.size(1)).zero_()

			if self.cuda:
				self.node_ranks[activation_index] = self.node_ranks[activation_index].cuda()

		#adding rank score for this batch of images to values from previous batch
		self.node_ranks[activation_index] += taylor
		self.grad_index += 1


	def normalize_node_ranks_per_layer(self):
		for i in self.node_ranks:
			v = torch.abs(self.node_ranks[i])
			v = v.cpu()
			v = v / np.sqrt(torch.sum(v * v))
			self.node_ranks[i] = v

	def format_node_ranks(self):
		formatted_rank = []
		unique_id = 0
		for i in sorted(self.node_ranks.keys()):
			for j in range(self.node_ranks[i].size(0)):
				formatted_rank.append((unique_id,i, j, float(self.node_ranks[i][j])))
				unique_id += 1
		return formatted_rank


