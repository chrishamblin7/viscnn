import torch
import os
import argparse
import sys

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("output_folder", type = str, help='the folder name for this prepped model')
	args = parser.parse_args()
	return args


args = get_args()
print(args)
output_folder = args.output_folder

sys.path.insert(0, os.path.abspath('../prepped_models/%s'%output_folder))

os.chdir(os.path.abspath('../prepped_models/%s'%output_folder))
import prep_model_params_used as params
os.chdir('../../prep_model_scripts')

model = params.model


def get_kernels_Conv2d_modules(module,kernels=[]): 
	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, torch.nn.modules.conv.Conv2d):
			kernels.append(submodule.weight.cpu().detach().numpy())
		elif len(list(submodule.children())) > 0:
			kernels = get_kernels_Conv2d_modules(submodule,kernels=kernels)   #module has modules inside it, so recurse on this module

	return kernels

kernels = get_kernels_Conv2d_modules(model)

torch.save(kernels,'../prepped_models/'+params.output_folder+'/kernels.pt')

