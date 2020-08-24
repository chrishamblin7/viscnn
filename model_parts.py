import torch
from torch.autograd import Variable
from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor

torch.set_default_tensor_type(torch.FloatTensor)

edgedect_filters = tensor([[[[-0.10, 0.0, 0.10],
                             [-0.20, 0.0, 0.20],
                             [-0.10, 0.0, 0.20]]],


                           [[[-0.10,-0.20,-0.10],
                             [ 0.00, 0.00, 0.00],
                             [ 0.10, 0.20, 0.20]]],


                           [[[-0.20,-0.10, 0.00],
                             [-0.10, 0.00, 0.10],
                             [ 0.00, 0.10, 0.20]]],


                           [[[ 0.00, 0.10, 0.00],
                             [ 0.10, 0.40, 0.10],
                             [ 0.00, 0.10, 0.00]]],


                           [[[ 0.10, 0.0, -0.10],
                             [ 0.20, 0.0, -0.20],
                             [ 0.10, 0.0, -0.20]]],


                           [[[ 0.10, 0.20, 0.10],
                             [ 0.00, 0.00, 0.00],
                             [-0.10,-0.20,-0.20]]]])

