#Get activations for single images
import torch
from PIL import Image
import os
import pickle
from torchvision import transforms


image_folder = '../data/cifar10/select'
labels = os.listdir(image_folder)
labels = sorted(labels)


model = torch.load('models/cifar/cifar_prunned_.816')
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform =   transforms.Compose([
                                 transforms.ToTensor(),
                                 normalize,
                             ])


def image_loader(image_name, transform=transform):
    image = Image.open(image_name)
    image = transform(image).float()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image.cuda()
    return image


'''
image_name = '../data/cifar10/select/airplane/0001.png'
image = image_loader(image_name)
print(image)
print(image.shape)
'''


output_dict = {}
for label in labels:
	print(label)
	output_dict[label] = {}
	image_names = os.listdir(os.path.join(image_folder,label))
	image_names = sorted(image_names)
	for image_name in image_names:
		print(image_name)
		x = image_loader(os.path.join(image_folder,label,image_name))
		activations = []
		for layer, (name, module) in enumerate(model.features._modules.items()):
			x = module(x)
			if isinstance(module, torch.nn.modules.conv.Conv2d):
				activations.append(x.squeeze().cpu().detach().numpy())
		output_dict[label][image_name] = activations


pickle.dump(output_dict,open('activations/cifar_prunned_.816_activations.pkl','wb'))
