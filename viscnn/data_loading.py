#functions for loading in image data
from PIL import Image, ImageOps
import os
from subprocess import call
import numpy as np
from torchvision import datasets, transforms, utils
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from random import randint


def single_image_loader(image_path, transform, label_file_path = None, label_dict_path = None):
	img_name = image_path.split('/')[-1].split('.')[:-1]
	if not transform:
		transform = transforms.Compose([transforms.ToTensor()])

	#get target (label)
	if label_dict_path is not None:
		label_dict = pickle.load(open(label_dict_path,'rb'))
		target = torch.tensor([9999999])
		if img_name in label_dict.keys():
			target = label_dict[img_name]
	else:
		if label_file_path is not None:
			label_file = open(label_file_path,'r')
			label_list = [x.strip() for x in label_file.readlines()]
			label_file.close()
			label_name = None
			label_num = None
			for i in range(len(label_list)): # see if any label in file name
				if label_list[i] in img_name:
					if label_name is None:
						label_name =  label_list[i]
						label_num = i
					elif len(label_list[i]) > len(label_name):
						label_name = label_list[i]
						label_num = i
		target = torch.tensor([9999999])
		if label_num is not None:
			target = torch.tensor([label_num])

	#get image	
	image = Image.open(image_path)
	image = transform(image).float()
	image = image.unsqueeze(0)

	return (image,target)



class rank_image_data(Dataset):

	def __init__(self, root_dir, transform, label_file_path = None, label_dict_path = None, class_folders=False ):
		
		
		self.root_dir = root_dir
		self.class_folders = class_folders
		if not self.class_folders:
			self.img_names = os.listdir(self.root_dir)
			self.img_names.sort()
		else:
			self.img_names = []
			self.classes = os.listdir(self.root_dir)
			for cl in self.classes:
				files = os.listdir(self.root_dir+'/'+cl)
				full_names = [cl+'/'+s for s in files]
				self.img_names += full_names


		self.label_list = None
		self.label_file_path = label_file_path
		if self.label_file_path is not None:
			label_file = open(label_file_path,'r')
			self.label_list = [x.strip() for x in label_file.readlines()]
			label_file.close()

		self.label_dict_path = label_dict_path
		self.label_dict = None
		if self.label_dict_path is not None:
			self.label_dict = pickle.load(open(label_dict_path,'rb'))

		if not transform:
			transform = transforms.Compose([transforms.ToTensor()])
		self.transform = transform

	def __len__(self):
		return len(self.img_names)

	def get_label_from_name(self,img_name):
		#check for label dict
		if self.label_dict is not None:
			if img_name not in self.label_dict.keys():
				return torch.tensor(9999999)
			else:
				return self.label_dict[img_name]
		else: #assume its a discrete one-hot label	
			if self.label_list is None:
				return torch.tensor(9999999)
			label_name = None
			label_num = None
			for i in range(len(self.label_list)): # see if any label in file name
				if self.label_list[i] in img_name:
					if label_name is None:
						label_name =  self.label_list[i]
						label_num = i
					elif len(self.label_list[i]) > len(label_name):
						label_name = self.label_list[i]
						label_num = i
			target = torch.tensor(9999999)
			if label_num is not None:
				target = torch.tensor(label_num)
			return target      

	def __getitem__(self, idx):

		img_path = os.path.join(self.root_dir,self.img_names[idx])
		img = Image.open(img_path)
		img = self.transform(img).float()
		label = self.get_label_from_name(self.img_names[idx])
		
		return (img,label)


def max_likelihood_for_no_target(target,model_output):
	pred = model_output.max(1, keepdim=True)[1].view_as(target)
	for i in range(len(target)):
		if target[i] == 9999999:
			target[i] = pred[i]
	return target

#out of use
def order_target(target,order_file):
	file = open(order_file,'r')
	reorder = [x.strip() for x in file.readlines()]
	current_order = deepcopy(reorder)
	current_order.sort()      #current order is alphabetical 
	if len(target.shape)==1:
		for i in range(len(target)):
			category_name = current_order[target[i]]
			target[i] = reorder.index(category_name)
		file.close()
		return target
	elif len(target.shape)==2:
		sys.exit('only 1 dimensional target vectors currently supported, not 2 :(')
	else:
		sys.exit('target has incompatible shape for reordering: %s'%str(target.shape))