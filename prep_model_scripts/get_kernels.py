import torch
import os
import argparse
import sys
import numpy as np

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

#function for return a kernels inhibition/exhitation value, normalized between -1 and 1
def gen_kernel_posneg(kernels):
    kernel_colors = []
    for i, layer in enumerate(kernels):
        average = np.average(np.average(layer,axis=3),axis=2)
        absum = np.sum(np.sum(np.abs(layer),axis=3),axis=2)
        unnormed_layer_colors = average/absum
        #normalize layer between -1 and 1
        normed_layer_colors = 2/(np.max(unnormed_layer_colors)-np.min(unnormed_layer_colors))*(unnormed_layer_colors-np.max(unnormed_layer_colors))+1
        kernel_colors.append(normed_layer_colors)
    return kernel_colors

#function that takes kernel posneg values from -1 to 1 and returns rgba values
def posneg_to_rgb(kernel_posneg,color_anchors = [[10, 87, 168],[170,170,170],[194, 0, 19]]):
    
    #define a function for converting 'p' values between 0 and 1 to a 3 color vector
    color_anchors = np.array(color_anchors)
    def f(p,color_anchors=color_anchors):
        if p < .5:
            return np.rint(np.minimum(np.array([255,255,255]),color_anchors[1] * p * 2 +  color_anchors[0] * (0.5 - p) * 2))
        else:
            return np.rint(np.minimum(np.array([255,255,255]),color_anchors[2] * (p - 0.5) * 2 +  color_anchors[1] * (1 - p) * 2))
    #fnp = np.frompyfunc(f,1,1) 
    fnp = np.vectorize(f,signature='()->(n)') 

    kernel_colors = []
    for i, layer in enumerate(kernel_posneg):
        #nonlinear color interpolation
        ps = (layer+1)/2
        #ps = 1/(1+np.exp(-2*layer))
        kernel_colors.append(fnp(ps))
    return kernel_colors



kernels = get_kernels_Conv2d_modules(model)
kernel_posneg = gen_kernel_posneg(kernels)
kernel_colors = posneg_to_rgb(kernel_posneg)

all_dict = {'kernels':kernels,
			'kernels_posneg':kernel_posneg,
			'kernel_colors':kernel_colors}

torch.save(all_dict,'../prepped_models/'+params.output_folder+'/kernels.pt')

