import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from model_parts import edgedect_filters

class cifar_CNN_prunned(nn.Module):
	"""CNN."""

	def __init__(self, out_channels = 10):
		"""CNN Builder."""
		super(cifar_CNN_prunned, self).__init__()

		self.out_channels = out_channels

		self.features = nn.Sequential(

			# Conv Layer block 1
			nn.Conv2d(in_channels=3, out_channels=45, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(in_channels=45, out_channels=23, kernel_size=3, padding=1, bias=False),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),

			# Conv Layer block 2
			nn.Conv2d(in_channels=23, out_channels=49, kernel_size=3, padding=1, bias=False),
			#nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(in_channels=49, out_channels=47, kernel_size=3, padding=1, bias=False),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=2, stride=2),
		)


		self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

		self.classifier = nn.Sequential(
			nn.Dropout(),
			nn.Linear(1692, 1000),
			nn.ReLU(),
			nn.Dropout(),
			nn.Linear(1000, 500),
			nn.ReLU(),
			nn.Linear(500, self.out_channels)
			)


	def forward(self, x):
		x = self.features(x)
		x = self.avgpool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


class MNIST(nn.Module):
	def __init__(self, out_channels = 10):
		super(MNIST, self).__init__()
		self.out_channels = out_channels

		self.conv1 = nn.Conv2d(1, 20, 5, 1)
		self.conv2 = nn.Conv2d(20, 50, 3, 1)
		self.avgpool = nn.AdaptiveAvgPool2d((4, 4))
		self.fc1 = nn.Linear(4*4*50, 500)
		self.fc2 = nn.Linear(500, self.out_channels)

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = self.avgpool(x)
		x = x.view(-1, 4*4*50)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class MNIST_edges(nn.Module):
	def __init__(self, out_channels = 10):
		super(MNIST_edges, self).__init__()
		self.out_channels = out_channels
		self.conv1 = nn.Conv2d(1, 6, 3, stride=1, padding=1, bias=False)
		self.conv2 = nn.Conv2d(6, 10, 3, stride=1, padding=1, bias=False)
		self.conv3 = nn.Conv2d(10, 14, 3, stride=1,padding=1, bias=False)
		self.fc1 = nn.Linear(28*28*14, 500)
		self.fc2 = nn.Linear(500, self.out_channels)

		self.conv1.weight.data = edgedect_filters

	def forward(self, x):
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.view(-1, 28*28*14)
		x = F.relu(self.fc1(x))
		x = self.fc2(x)
		x = F.log_softmax(x, dim=1)
		return x

class AlexNet_format(nn.Module):

    def __init__(self, num_classes=1000):
        super(AlexNet_format, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x