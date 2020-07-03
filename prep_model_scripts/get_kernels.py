import torch
import parameters as params

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

