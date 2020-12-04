import torch
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import pdb
import time

#Error classes for breaking forward pass of model
# define Python user-defined exceptions
class ModelBreak(Exception):
	"""Base class for other exceptions"""
	pass

class TargetReached(ModelBreak):
	"""Raised when the output target for a subgraph is reached, so the model doesnt neeed to be run forward any farther"""
	#print('weird')
	pass



class dissected_Conv2d(torch.nn.Module):       #2d conv Module class that has presum activation maps as intermediate output
	
	def gen_inout_permutation(self):
		'''
		When we flatten out all the output channels not to be grouped by 'output channel', we still want the outputs sorted
		such that they can be conveniently added based on input channel later
		'''
		in_chan = self.in_channels
		out_chan = self.out_channels
		
		weight_perm = []
		for i in range(in_chan):
			for j in range(out_chan):
				weight_perm.append(i+j*in_chan)
		
		add_perm = []
		add_indices = {}
		for o in range(out_chan):
			add_indices[o] = []
			for i in range(in_chan):
				add_perm.append(o+i*out_chan)
				add_indices[o].append(o+i*out_chan)
		return torch.LongTensor(weight_perm),torch.LongTensor(add_perm),add_indices


	def make_preadd_conv(self):
		'''
		nn.Conv2d takes in 'in_channel' number of feature maps, and outputs 'out_channel' number of maps. 
		internally it has in_channel*out_channel number of 2d conv kernels. Normally, featuremaps associated 
		with a particular output channel resultant from these kernel convolution are all added together,
		this function changes a nn.Conv2d module into a module where this final addition doesnt happen. 
		The final addition can be performed seperately with permute_add_feature_maps.
		'''
		in_chan = self.in_channels
		out_chan = self.out_channels
		
		kernel_size = self.from_conv.kernel_size
		padding = self.from_conv.padding
		stride = self.from_conv.stride
		new_conv = nn.Conv2d(in_chan,in_chan*out_chan,kernel_size = kernel_size,
							 bias = False, padding=padding,stride=stride,groups= in_chan)
		new_conv.weight = torch.nn.parameter.Parameter(
				self.from_conv.weight.view(in_chan*out_chan,1,kernel_size[0],kernel_size[1])[self.weight_perm])
		return new_conv

		
	def permute_add_featuremaps(self,feature_map):
		'''
		Perform the sum within output channels step.  (THIS NEEDS TO BE SPEED OPTIMIZED)
		'''
		x = feature_map
		x = x[:, self.add_perm, :, :]
		x = torch.split(x.unsqueeze(dim=1),self.in_channels,dim = 2)
		x = torch.cat(x,dim = 1)
		x = torch.sum(x,dim=2)
		return x
	
	def gen_weight_ranks(self):
		weight_ranks_flat = torch.abs(self.preadd_conv.weight).mean(dim=(2,3)).data.squeeze(1)
		edge_weight_ranks = []
		for o in self.add_indices:
			in_chans = []
			for i in self.add_indices[o]:
				in_chans.append(weight_ranks_flat[i])
			edge_weight_ranks.append(in_chans)
		edge_weight_ranks = torch.tensor(edge_weight_ranks)
		node_weight_ranks = edge_weight_ranks.mean(dim=1)
		return weight_ranks_flat, node_weight_ranks


	
	def __init__(self, from_conv,name,store_activations=False, store_ranks = False, clear_ranks=False, target_node=None, cuda=True):      # from conv is normal nn.Conv2d object to pull weights and bias from
		super(dissected_Conv2d, self).__init__()
		self.name = name
		self.from_conv = from_conv
		self.in_channels = self.from_conv.weight.shape[1]
		self.out_channels = self.from_conv.weight.shape[0]
		self.target_node= target_node
		self.cuda = cuda
		self.store_activations = store_activations
		self.store_ranks = store_ranks
		self.clear_ranks = clear_ranks

		self.postbias_ranks_prenorm = {'act':None,'grad':None,'actxgrad':None}
		self.preadd_ranks_prenorm = {'act':None,'grad':None,'actxgrad':None}
		for rank_type in ['act','grad','actxgrad']:
			self.postbias_ranks_prenorm[rank_type] = torch.FloatTensor(self.out_channels).zero_()
			self.preadd_ranks_prenorm[rank_type] = torch.FloatTensor(self.out_channels*self.in_channels).zero_()
			if self.cuda:
				self.postbias_ranks_prenorm[rank_type] = self.postbias_ranks_prenorm[rank_type].cuda()
				self.preadd_ranks_prenorm[rank_type] = self.preadd_ranks_prenorm[rank_type].cuda()

		self.images_seen = 0
		self.weight_perm,self.add_perm,self.add_indices = self.gen_inout_permutation()
		self.preadd_conv = self.make_preadd_conv()
		self.bias = None
		if self.from_conv.bias is not None:
			self.bias = from_conv.bias.unsqueeze(1).unsqueeze(1)
			if self.cuda:
				self.bias = self.bias.cuda()
		#generate a dict that says which indices should be added together in for 'permute_add_featuremaps'

		self.preadd_ranks_prenorm['weight'],self.postbias_ranks_prenorm['weight'] = self.gen_weight_ranks()
		if self.store_ranks:
			self.preadd_out_hook = None
			self.postbias_out_hook = None

	def compute_edge_rank(self,grad):
		activation = self.preadd_out
		#activation_relu = F.relu(activation)
		taylor = activation * grad 
		rank_key  = {'act':torch.abs(activation),'grad':torch.abs(grad),'actxgrad':torch.abs(taylor)}
		for key in rank_key:
			if self.preadd_ranks_prenorm[key] is None: #initialize at 0
				self.preadd_ranks_prenorm[key] = torch.FloatTensor(activation.size(1)).zero_()
				if self.cuda:
					self.preadd_ranks_prenorm[key] = self.preadd_ranks_prenorm[key].cuda()
			map_mean = rank_key[key].mean(dim=(2, 3)).data
			mean_sum = map_mean.sum(dim=0).data      
			self.preadd_ranks_prenorm[key] += mean_sum    # we sum up the mean activations over all images, after all batches

		#print('length edge_rank: ', len(self.preadd_ranks_prenorm['actxgrad']))
		#print('length outdimxindim: ', self.out_channels*self.in_channels)
			#have passed through we will average by the number of images seen with self.average_ranks
		#print('edge_rank time: %s'%str(time.time() - start))



	def compute_node_rank(self,grad):
		activation = self.postbias_out
		activation_relu = F.relu(activation)
		taylor = activation_relu * grad 
		rank_key  = {'act':activation_relu,'grad':torch.abs(grad),'actxgrad':taylor}
		for key in rank_key:
			if self.postbias_ranks_prenorm[key] is None: #initialize at 0
				self.postbias_ranks_prenorm[key] = torch.FloatTensor(activation.size(1)).zero_()
				if self.cuda:
					self.postbias_ranks_prenorm[key] = self.postbias_ranks_prenorm[key].cuda()
			map_mean = rank_key[key].mean(dim=(2, 3)).data
			mean_sum = map_mean.sum(dim=0).data      
			self.postbias_ranks_prenorm[key] += mean_sum    # we sum up the mean activations over all images, after all batches
			#have passed through we will average by the number of images seen with self.average_ranks
		#print('length node_rank: ', len(self.postbias_ranks_prenorm['actxgrad']))
		#print('length outdim: ', self.out_channels)
		#print('node_rank time: %s'%str(time.time() - start))



	def format_edges(self, data= 'activations',prenorm=False,weight_rank=False):
		#fetch preadd activations as [img,out_channel, in_channel,h,w]
		#fetch preadd ranks as [out_chan,in_chan]
		import pdb;pdb.set_trace()
		if weight_rank:
			rank_types = ['weight']
		else:
			rank_types = ['act','grad','actxgrad']

		if not self.store_activations:
			print('activations arent stored, use "store_activations=True" on model init. returning None')
			return None
		out_acts_list = []
		if data == 'activations':
			for out_chan in self.add_indices:
				in_acts_list = []
				for in_chan in self.add_indices[out_chan]:
					in_acts_list.append(self.preadd_out[:,in_chan,:,:].unsqueeze(dim=1).unsqueeze(dim=1))                    
				out_acts_list.append(torch.cat(in_acts_list,dim=2))
			return torch.cat(out_acts_list,dim=1).cpu().detach().numpy()

		else:
			output = {}
			for rank_type in rank_types:
				out_acts_list = []
				for out_chan in self.add_indices:
					in_acts_list = []
					for in_chan in self.add_indices[out_chan]:
						if not prenorm:
							in_acts_list.append(self.preadd_ranks[rank_type][in_chan].unsqueeze(dim=0).unsqueeze(dim=0)) 
						else:
							in_acts_list.append(self.preadd_ranks_prenorm[rank_type][in_chan].unsqueeze(dim=0).unsqueeze(dim=0))                 
					out_acts_list.append(torch.cat(in_acts_list,dim=1))
				output[rank_type] = torch.cat(out_acts_list,dim=0).cpu().detach().numpy()
			return output
						
	def average_ranks(self):
		for rank_type in ['act','grad','actxgrad']:
			if self.images_seen > 0:
				self.preadd_ranks_prenorm[rank_type] = self.preadd_ranks_prenorm[rank_type]/self.images_seen
				self.postbias_ranks_prenorm[rank_type] = self.postbias_ranks_prenorm[rank_type]/self.images_seen

	def abs_ranks(self):
		for rank_type in ['act','grad','actxgrad']:
			self.preadd_ranks_prenorm[rank_type] = torch.abs(self.preadd_ranks_prenorm[rank_type])
			self.postbias_ranks_prenorm[rank_type] = torch.abs(self.postbias_ranks_prenorm[rank_type])       

	def normalize_ranks(self):
		self.preadd_ranks = {}
		self.postbias_ranks = {}
		for rank_type in ['act','grad','actxgrad','weight']:
			if self.images_seen > 0:
				e = torch.abs(self.preadd_ranks_prenorm[rank_type])
				n = torch.abs(self.postbias_ranks_prenorm[rank_type])

				e = e.cpu()
				e = e / np.sqrt(torch.sum(e * e))
				n = n.cpu()
				n = n / np.sqrt(torch.sum(n * n))

				self.preadd_ranks[rank_type] = e
				self.postbias_ranks[rank_type] = n
			else:
				self.preadd_ranks[rank_type] = self.preadd_ranks_prenorm[rank_type]
				self.postbias_ranks[rank_type] = self.postbias_ranks_prenorm[rank_type]


		#self.preadd_ranks_prenorm['weight'] = self.preadd_ranks_prenorm['weight'].cpu()
		#self.postbias_ranks_prenorm['weight'] = self.postbias_ranks_prenorm['weight'].cpu()
		#self.preadd_ranks['weight'] = torch.abs(self.preadd_ranks_prenorm['weight'] )/np.sqrt(torch.sum(self.preadd_ranks_prenorm['weight'] *self.preadd_ranks_prenorm['weight'] ))
		#self.postbias_ranks['weight'] = torch.abs(self.postbias_ranks_prenorm['weight'])/np.sqrt(torch.sum(self.postbias_ranks_prenorm['weight']*self.postbias_ranks_prenorm['weight']))

	def clear_ranks_func(self): #clear ranks, info that otherwise accumulates with images
		self.images_seen = 0
		for rank_type in ['act','grad','actxgrad']:
			self.postbias_ranks_prenorm[rank_type] = torch.FloatTensor(self.out_channels).zero_()
			self.preadd_ranks_prenorm[rank_type] = torch.FloatTensor(self.out_channels*self.in_channels).zero_()
			if self.cuda:
				self.postbias_ranks_prenorm[rank_type] = self.postbias_ranks_prenorm[rank_type].cuda()
				self.preadd_ranks_prenorm[rank_type] = self.preadd_ranks_prenorm[rank_type].cuda()

	def forward(self, x):
		
		if self.clear_ranks:
			self.clear_ranks_func()

		self.images_seen += x.shape[0]    #keep track of how many images weve seen so we know what to divide by when we average ranks
		if self.store_activations:
			self.input = x

		preadd_out = self.preadd_conv(x)  #get output of convolutions

		#store values of intermediate outputs after convolution
		if self.store_activations:
			self.preadd_out = preadd_out
 
		#Set hooks for calculating rank on backward pass
		if self.store_ranks:
			self.preadd_out = preadd_out
			if self.preadd_out_hook is not None:
				self.preadd_out_hook.remove()
			self.preadd_out_hook = self.preadd_out.register_hook(self.compute_edge_rank)
			#if self.preadd_ranks is not None:
			#    print(self.preadd_ranks.shape)

		added_out = self.permute_add_featuremaps(preadd_out)    #add convolution outputs by output channel
		if self.bias is not None:  
			postbias_out = added_out + self.bias
		else:
			postbias_out = added_out

		#Store values of final module output
		if self.store_activations:
			self.postbias_out = postbias_out
 
		#Set hooks for calculating rank on backward pass
		if self.store_ranks:
			self.postbias_out = postbias_out
			if self.postbias_out_hook is not None:
				self.postbias_out_hook.remove()
			self.postbias_out_hook = self.postbias_out.register_hook(self.compute_node_rank)
			#if self.postbias_ranks is not None:
			#    print(self.postbias_ranks.shape)
		
		if self.target_node is not None:
			print('target node reached, stopping forward pass')
			print(self.target_node)
			avg_activations = self.postbias_out.mean(dim=(0, 2, 3))
			optim_target = avg_activations[self.target_node]
			print(optim_target)
			optim_target.backward()
			raise TargetReached
			
		
		return postbias_out



# takes a full model and replaces all conv2d instances with dissected conv 2d instances
def dissect_model(model,mod_names = [],store_activations=True,store_ranks=True,clear_ranks = False,cuda=True):

	for name, module in reversed(model._modules.items()):
		if len(list(module.children())) > 0:
			mod_names.append(str(name))
			# recurse
			model._modules[name] = dissect_model(module,mod_names =mod_names, store_activations=store_activations,store_ranks=store_ranks,clear_ranks=clear_ranks,cuda=cuda)
			mod_names.pop()

		if isinstance(module, torch.nn.modules.conv.Conv2d):    # found a 2d conv module to transform
			new_module = dissected_Conv2d(module, name='_'.join(mod_names+[name]), store_activations=store_activations,store_ranks=store_ranks,clear_ranks=clear_ranks,cuda=cuda) 
			model._modules[name] = new_module

		elif isinstance(module, torch.nn.modules.Dropout):    #make dropout layers not dropout  #also set batchnorm to eval
			model._modules[name].eval() 


		else:    #make activation functions not 'inplace'
			model._modules[name].inplace=False                    

	return model


def set_model_target_node(model,target_layer,within_layer_id,layer=0):
 
	for name, module in model._modules.items():
		if len(list(module.children())) > 0:
			# recurse
			model._modules[name] = set_model_target_node(module,target_layer,within_layer_id,layer)

		if isinstance(module, dissected_Conv2d):    # found a 2d conv module to transform
			if layer==target_layer:
				module.target_node = within_layer_id
				break
			layer+=1             

	return model

def set_across_model(model,setting,value):
 
	for name, module in model._modules.items():
		if len(list(module.children())) > 0:
			# recurse
			model._modules[name] = set_across_model(module,setting,value)

		if isinstance(module, dissected_Conv2d):    # found a 2d conv module to transform
			if setting == 'target_node':
				module.target_node = value
			if setting == 'clear_ranks':
				module.clear_ranks=value
			 

	return model


def clear_ranks_across_model(model):
 
	for name, module in model._modules.items():
		if len(list(module.children())) > 0:
			# recurse
			model._modules[name] = clear_ranks_across_model(module)

		if isinstance(module, dissected_Conv2d):    # found a 2d conv module to transform
			module.clear_ranks_func()
			 

	return model

