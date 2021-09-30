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
from copy import deepcopy
from torchvision import models, transforms
import torch.utils.data as data
import torchvision.datasets as datasets

model = models.alexnet(pretrained=True)
model.cuda()



normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
				transforms.CenterCrop(224),
				transforms.ToTensor(),
				normalize
				])


image_loader = data.DataLoader(
					datasets.ImageFolder('/mnt/data/chris/dropbox/Research-Hamblin/Projects/cnn_subgraph_visualizer/image_data/imagenet_200/ranking_images/', preprocess),
					batch_size=200,
					shuffle=True,
					num_workers=4,
					pin_memory=True)



def order_target(target,order_file):
	file = open(order_file,'r')
	reorder = [x.strip() for x in file.readlines()]
	current_order = deepcopy(reorder)
	current_order.sort()      #current order is alphabetical 
	if len(target.shape)==1:
		for i in range(len(target)):
			class_name = current_order[target[i]]
			target[i] = reorder.index(class_name)
		return target
	elif len(target.shape)==2:
		sys.exit('only 1 dimensional target vectors currently supported, not 2 :(')
	else:
		sys.exit('target has incompatible shape for reordering: %s'%str(target.shape))


#Pass data through model in batches
for i, (batch, target) in enumerate(image_loader):
	print('batch %s'%i)
	batch = Variable(batch)
	batch = batch.cuda()
	target = order_target(target,'/mnt/data/chris/dropbox/Research-Hamblin/Projects/cnn_subgraph_visualizer/image_data/imagenet_200/ranking_images/label_order.txt')
	target = target.cuda()
	output = model(batch)    #running forward pass sets up hooks and stores activations in each dissected_Conv2d module
	if i == 0 or i == 1:
		break
	#try:
	#	params.criterion(output, Variable(target)).backward()    #running backward pass calls all the hooks and calculates the ranks of all edges and nodes in the graph 
	#except:
	#	torch.sum(output).backward()    # run backward pass with respect to net outputs rather than loss function

print('output')
print(output.shape)
print(output)

print('target')
print(target.shape)
print(target)


pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
correct = pred.eq(target.view_as(pred)).sum().item()

print(correct)