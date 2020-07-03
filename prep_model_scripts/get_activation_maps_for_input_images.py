#Get activations for single images
import torch
from PIL import Image, ImageOps
import os
import numpy as np
from torchvision import datasets, transforms, utils
from dissected_Conv2d import *
import parameters as params
from copy import deepcopy


###MODEL LOADING
model=params.model 
if params.cuda:
	model = model.cuda()
else:
	model = model.cpu()
model_dis = dissect_model(deepcopy(model),store_ranks=False,cuda = params.cuda)

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
		layer_activations = {'nodes':[],'edges':[]}
	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, dissected_Conv2d):
			layer_activations['nodes'].append(submodule.postbias_out.cpu().detach().numpy())
			layer_activations['edges'].append(submodule.format_edges(data= 'activations'))
			print(layer_activations['edges'][-1].shape)
		elif len(list(submodule.children())) > 0:
			layer_activations = get_activations_from_dissected_Conv2d_modules(submodule,layer_activations=layer_activations)   #module has modules inside it, so recurse on this module

	return layer_activations

layer_activations = get_activations_from_dissected_Conv2d_modules(model_dis)
print(model_dis)


###SAVE OUTPUT
torch.save(layer_activations, '../prepped_models/'+params.output_folder+'/input_img_activations.pt')
