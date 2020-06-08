#Get activations for single images
import torch
from PIL import Image, ImageOps
import os
import pickle
import numpy as np
from torchvision import datasets, transforms, utils
from dissected_Conv2d import *
import parameters as params
from copy import deepcopy

image_folder = params.input_img_path
image_names = os.listdir(image_folder)
image_names = sorted(image_names)




#model loading
model=params.model 
model.eval()

transform = params.preprocess

def image_loader(image_folder = image_folder, transform=transform):
	img_names = os.listdir(image_folder)
	img_names.sort()
	image_list = []
	for img_name in img_names:
		image = Image.open(os.path.join(image_folder,img_name))
		image = transform(image).float()
		#image = transforms.ToTensor(image)    #make sure image is float tensor
		#image = torch.tensor(image, requires_grad=True)
		image = image.unsqueeze(0)
		if params.cuda:
			image = image.cuda()
		image_list.append(image)
	return torch.cat(image_list,0)

images = image_loader()

output_dict = {}

print('getting node activation maps\n')

x = images
activations = []

def get_node_activations_recursive(x,module):
	for layer, (name, submodule) in enumerate(module._modules.items()):
		if len(list(submodule.children())) > 0:
			x = get_node_activations_recursive(x,submodule)
		else:
			try:
				#print(submodule)
				x = submodule(x)
				if isinstance(submodule, torch.nn.modules.conv.Conv2d):
					#print('found conv2d')
					activations.append(x.cpu().detach().numpy())
			except:
				print('cant continue without specifying forward model')
				break
	return x

x = get_node_activations_recursive(x,model)

'''
for layer, (name, module) in enumerate(model._modules.items()):
	try:
		print(module)
		x = module(x)
		if isinstance(module, torch.nn.modules.conv.Conv2d):
			activations.append(x.cpu().detach().numpy())
	except:
		print('cant continue without specifying forward model')
		break
'''
output_dict['nodes'] = activations

print('\ngetting edge activation maps\n')

x = images
activations = []
dis_model = dissect_model(deepcopy(model),store_activations=True)
out = dis_model(x)

def get_edge_activations_recursive(x,module):
	for layer, (name, submodule) in enumerate(module._modules.items()):
		if len(list(submodule.children())) > 0:
			get_edge_activations_recursive(x,submodule)
		else:
			if isinstance(submodule, dissected_Conv2d):
				activations.append(submodule.activations.cpu().detach().numpy())

get_edge_activations_recursive(x,model)

'''
for layer, (name, module) in enumerate(dis_model._modules.items()):
	if isinstance(module, dissected_Conv2d):
		#print(module.add_indices)
		activations.append(module.activations.cpu().detach().numpy())
'''

output_dict['edges'] = activations




pickle.dump(output_dict,open('activations/cifar_prunned_.816_all_activations.pkl','wb'))