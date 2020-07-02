import torch
import pickle



model = torch.load('models/cifar/cifar_prunned_.816')



kernels = {}
i=0
for layer, (name, module) in enumerate(model.features._modules.items()):
	if isinstance(module, torch.nn.modules.conv.Conv2d):
		kernels[i] = model.features[layer].weight.cpu().detach().numpy()
		i+=1

pickle.dump(kernels,open('kernels/cifar_prunned_.816_kernels.pkl','wb'))