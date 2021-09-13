import time
import torch
from subprocess import call
import os
from dissected_Conv2d import *
from torch.autograd import Variable
from copy import deepcopy
import pandas as pd
import pickle
import sys
sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../visualizer_scripts/'))
from visualizer_helper_functions import get_ranks_from_dissected_Conv2d_modules
from featureviz_helper_functions import *

import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("output_folder", type = str, help='the folder name for this prepped model')
	parser.add_argument("--device", type = str, default='none', help='device to run process on,  (cuda:0, cuda:1, cpu etc) (defaults to prep_model params device)')
	args = parser.parse_args()
	return args

args = get_args()
output_folder = args.output_folder
prepped_model_folder = output_folder

full_prepped_model_folder = os.path.abspath('../prepped_models/%s'%prepped_model_folder)

sys.path.insert(0,full_prepped_model_folder)


sys.path.insert(0, os.path.abspath('../prepped_models/%s'%output_folder))

os.chdir(os.path.abspath('../prepped_models/%s'%output_folder))
from prep_model_params_used import deepviz_param, deepviz_optim, deepviz_transforms, deepviz_image_size, deepviz_neuron
import prep_model_params_used as prep_model_params
os.chdir('../../prep_model_scripts')


params = {}
params['prepped_model'] = prepped_model_folder
params['prepped_model_path'] = full_prepped_model_folder
#Non-GUI parameters

#deepviz
#params['deepviz_param'] = None
#params['deepviz_optim'] = None
#params['deepviz_transforms'] = None

params['deepviz_param'] = deepviz_param
params['deepviz_optim'] = deepviz_optim
params['deepviz_transforms'] = deepviz_transforms
params['deepviz_image_size'] = deepviz_image_size
params['deepviz_neuron'] = deepviz_neuron

#backend
params['cuda'] = prep_model_params.cuda    #use gpu acceleration when running model forward
params['device'] = prep_model_params.device
params['input_image_directory'] = prep_model_params.input_img_path+'/'   #path to directory of imput images you want fed through the network
params['preprocess'] = prep_model_params.preprocess     #torchvision transfrom to pass input images through
params['label_file_path'] = prep_model_params.label_file_path
params['criterion'] = prep_model_params.criterion
params['rank_img_path'] = prep_model_params.rank_img_path
params['num_workers'] = prep_model_params.num_workers
params['seed'] = prep_model_params.seed
params['batch_size'] = prep_model_params.batch_size



#load misc graph data
print('loading misc graph data')
misc_data = pickle.load(open(full_prepped_model_folder+'/misc_graph_data.pkl','rb'))
params['layer_nodes'] = misc_data['layer_nodes']
params['num_layers'] = misc_data['num_layers']
params['num_nodes'] = misc_data['num_nodes']
params['categories'] = misc_data['categories']
params['num_img_chan'] = misc_data['num_img_chan']
params['imgnode_positions'] = misc_data['imgnode_positions']
params['imgnode_colors'] = misc_data['imgnode_colors']
params['imgnode_names'] = misc_data['imgnode_names']
params['prepped_model_path'] = full_prepped_model_folder
params['ranks_data_path'] = full_prepped_model_folder+'/ranks/'

if not args.device == 'none':
	params['device'] = args.device


##MODEL LOADING
model = prep_model_params.model
if prep_model_params.cuda:
	model.cuda()
model.to(params['device'])
model = model.eval()    

if params['deepviz_neuron']:

	model_dis = dissect_model(deepcopy(prep_model_params.model),store_ranks=True,clear_ranks=True,cuda=params['cuda'],device=params['device']) #version of model with accessible preadd activations in Conv2d modules 
	if params['cuda']:
		model_dis.cuda()
	model_dis = model_dis.eval()    
	model_dis.to(params['device'])
	model = model_dis

print('loaded model:')
print(prep_model_params.model)


for param in model.parameters():  #need gradients for grad*activation rank calculation
	param.requires_grad = True
#os.chdir('./prep_model_scripts')



start = time.time()

torch.manual_seed(params['seed'])


print('generating deep vizualizations for nodes')
for nodeid in range(params['num_nodes']):
	print(nodeid)
	image_name = fetch_deepviz_img(model,str(nodeid),params)