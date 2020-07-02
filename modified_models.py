import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F




		

class CustomNetModel(torch.nn.Module):
	def __init__(self, out_channels = 46, init_weights = False):
		super(CustomNetModel, self).__init__()

		self.out_channels = out_channels

		self.features = nn.Sequential(
			nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True),
			#nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			#nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			#nn.ReLU(inplace=True),
			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			nn.ReLU(inplace=True))
			#nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			#nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			#nn.ReLU(inplace=True))


		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512*4*4, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 1028),
			nn.ReLU(inplace=True),
			nn.Linear(1028, self.out_channels))

		if init_weights:
			self.features.apply(init_weights)
			self.classifier.apply(init_weights)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x



class CustomNetModel_nobatchnorm(torch.nn.Module):
	def __init__(self, out_channels = 46, init_weights = False):
		super(CustomNetModel_nobatchnorm, self).__init__()

		self.out_channels = out_channels

		self.features = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			#nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			#nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			#nn.ReLU(inplace=True),
			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True))
			#nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			#nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			#nn.ReLU(inplace=True))


		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(512*4*4, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 1028),
			nn.ReLU(inplace=True),
			nn.Linear(1028, self.out_channels))

		if init_weights:
			self.features.apply(init_weights)
			self.classifier.apply(init_weights)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x

class CustomNetModel_trained_architecture(torch.nn.Module):
	def __init__(self, out_channels = 46, init_weights = False):
		super(CustomNetModel_trained_architecture, self).__init__()
		self.out_channels = out_channels

		self.features = nn.Sequential(
			nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
			nn.Conv2d(64, 23, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(23, 56, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(56, 41, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(41, 103, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			#nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			#nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			#nn.ReLU(inplace=True),
			nn.Conv2d(103, 95, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True))
			#nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			#nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
			#nn.ReLU(inplace=True))


		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(95*4*4, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 1028),
			nn.ReLU(inplace=True),
			nn.Linear(1028, self.out_channels))

		if init_weights:
			self.features.apply(init_weights)
			self.classifier.apply(init_weights)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


class CustomNetModel_small(torch.nn.Module):
	def __init__(self, out_channels = 46, init_weights = False):
		super(CustomNetModel_small, self).__init__()
		self.out_channels = out_channels

		self.features = nn.Sequential(
			nn.Conv2d(1, 50, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
			nn.Conv2d(50, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(20, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(50, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True))


		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(100*4*4, 1000),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(1000, 500),
			nn.ReLU(inplace=True),
			nn.Linear(500, self.out_channels))

		if init_weights:
			self.features.apply(init_weights)
			self.classifier.apply(init_weights)

class cifar_CNN(nn.Module):
	"""CNN."""


	def __init__(self, out_channels = 10, init_weights = False):
		"""CNN Builder."""
		super(cifar_CNN, self).__init__()

		self.out_channels = out_channels

		self.features = nn.Sequential(

			# Conv Layer block 1
			nn.Conv2d(in_channels=3, out_channels=100, kernel_size=3, padding=1),
			#nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=100, out_channels=50, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			# Conv Layer block 2
			nn.Conv2d(in_channels=50, out_channels=100, kernel_size=3, padding=1),
			#nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=100, out_channels=100, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)


		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(100*6*6, 1000),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(1000, 500),
			nn.ReLU(inplace=True),
			nn.Linear(500, self.out_channels)
			)

		
		

		if init_weights:
			self.features.apply(self.weight_init)
			self.classifier.apply(self.weight_init)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x




class CustomNetModel_small_nopool(torch.nn.Module):
	def __init__(self, out_channels = 46, init_weights = False):
		super(CustomNetModel_small_nopool, self).__init__()
		self.out_channels = out_channels

		self.features = nn.Sequential(
			nn.Conv2d(1, 50, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(50, 20, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(20, 50, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True),
			nn.Conv2d(50, 100, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
			nn.ReLU(inplace=True))


		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(100*4*4, 1000),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(1000, 500),
			nn.ReLU(inplace=True),
			nn.Linear(500, self.out_channels))

		if init_weights:
			self.features.apply(init_weights)
			self.classifier.apply(init_weights)

	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


class ModifiedVGG16Model(torch.nn.Module):
	def __init__(self, out_channels = 1000):
		super(ModifiedVGG16Model, self).__init__()

		model = models.vgg16(pretrained=True)
		self.features = model.features
		self.out_channels = out_channels
		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(25088, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, self.out_channels))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x


class ModifiedAlexnetModel(torch.nn.Module):
	def __init__(self, out_channels = 1000):
		super(ModifiedAlexnetModel, self).__init__()

		model = models.alexnet(pretrained=True)
		self.features = model.features
		self.out_channels = out_channels
		#animate out 395
		#inanimate out 605
		for param in self.features.parameters():
			param.requires_grad = False

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(9216, 4096),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(4096, 4096),
			nn.ReLU(inplace=True),
			nn.Linear(4096, self.out_channels))

	def forward(self, x):
		x = self.features(x)
		x = x.view(x.size(0), -1)
		x = self.classifier(x)
		return x