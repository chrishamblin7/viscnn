#Get activations for single images
import torch
from PIL import Image, ImageOps
import os
import numpy as np
from torchvision import datasets, transforms, utils
from dissected_Conv2d import *
from copy import deepcopy
import sys
sys.path.insert(0, os.path.abspath('../'))

os.chdir('../')
import prep_model_parameters as params
os.chdir('./prep_model_scripts')

###MODEL LOADING

model_dis = dissect_model(deepcopy(params.model),store_ranks=False,cuda=params.cuda) #version of model with accessible preadd activations in Conv2d modules 
if params.cuda:
	model_dis.cuda()
del params.model

###IMAGE LOADING
image_folder = params.input_img_path
image_names = os.listdir(image_folder)
image_names = sorted(image_names)

def image_loader(image_folder = image_folder, transform=params.preprocess):
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


###RUN MODEL
output = model_dis(images) #run model forward, storing activations

#run through all model modules recursively, and pull the activations stored in dissected_Conv2d modules 
def get_activations_from_dissected_Conv2d_modules(module,layer_activations=None):     
	if layer_activations is None:    #initialize the output dictionary if we are not recursing and havent done so yet
		layer_activations = {'nodes':[],'edges_in':[],'edges_out':[]}
	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, dissected_Conv2d):
			layer_activations['nodes'].append(submodule.postbias_out.cpu().detach().numpy())
			layer_activations['edges_in'].append(submodule.input.cpu().detach().numpy())
			layer_activations['edges_out'].append(submodule.format_edges(data= 'activations'))
			print(layer_activations['edges_out'][-1].shape)
		elif len(list(submodule.children())) > 0:
			layer_activations = get_activations_from_dissected_Conv2d_modules(submodule,layer_activations=layer_activations)   #module has modules inside it, so recurse on this module

	return layer_activations

layer_activations = get_activations_from_dissected_Conv2d_modules(model_dis)

#reformat activations so images dont take up a dimension in the np array, 
# but rather there is an individual array for each image name key in a dict
def act_array_2_imgname_dict(layer_activations, image_names):
	new_activations = {'nodes':{},'edges_in':{},'edges_out':{}}
	for i in range(len(image_names)):
		for part in ['nodes','edges_in','edges_out']:
			new_activations[part][image_names[i]] = []
			for l in range(len(layer_activations[part])):
				new_activations[part][image_names[i]].append(layer_activations[part][l][i])
	return new_activations
			
layer_activations = act_array_2_imgname_dict(layer_activations,image_names)

###SAVE OUTPUT
torch.save(layer_activations, '../prepped_models/'+params.output_folder+'/input_img_activations.pt')
