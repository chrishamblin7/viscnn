import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F


class cifar_CNN_prunned(nn.Module):
	"""CNN."""

	def __init__(self, out_channels = 10):
		"""CNN Builder."""
		super(cifar_CNN_prunned, self).__init__()

		self.out_channels = out_channels

		self.features = nn.Sequential(

			# Conv Layer block 1
			nn.Conv2d(in_channels=3, out_channels=45, kernel_size=3, padding=1),
			#nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=45, out_channels=23, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),

			# Conv Layer block 2
			nn.Conv2d(in_channels=23, out_channels=49, kernel_size=3, padding=1),
			#nn.BatchNorm2d(64),
			nn.ReLU(inplace=True),
			nn.Conv2d(in_channels=49, out_channels=47, kernel_size=3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)


		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(1692, 1000),
			nn.ReLU(inplace=True),
			nn.Dropout(),
			nn.Linear(1000, 500),
			nn.ReLU(inplace=True),
			nn.Linear(500, self.out_channels)
			)


	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x
