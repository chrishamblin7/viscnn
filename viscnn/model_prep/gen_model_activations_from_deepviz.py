#This script generates the deepviz_activations.pkl file, by running the deepviz images through themodel.
#Which is used for getting projections that use deepviz activations as the basis for each node
# #THIS IS CURRENTLY NOT WELL OPTIMIZED AND RUNS SLOW 
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

model_dis = dissect_model(deepcopy(prep_model_params.model),store_ranks=True,clear_ranks=True,cuda=params['cuda'],device=params['device']) #version of model with accessible preadd activations in Conv2d modules 
if params['cuda']:
    model_dis.cuda()
model_dis = model_dis.eval()    
model_dis.to(params['device'])
model = model_dis

##This function is horribly unoptimized, should not have to run through dissected model, and certainly should have to run 
##one image at a time
def gen_deepviz_activations(model_dis,params):
    type_names = {True:'neuron',False:'channel'}

    cuda = params['cuda']
    model_dis = set_across_model(model_dis,'target_node',None)
    if 'device' in params.keys():
        model_dis.to(params['device'])
    activations = []
    
    file_path = params['prepped_model_path']+'/visualizations/images.csv'
    images_df = pd.read_csv(file_path,dtype=str)
    for layer_num in range(len(params['layer_nodes'])):
        #initialize activation arrays to zeros
        activations.append(np.zeros([len(params['layer_nodes'][layer_num][1]),len(params['layer_nodes'][layer_num][1])]))
        for within_layer_nodeid, nodeid in enumerate(params['layer_nodes'][layer_num][1]):
            objective_str = gen_objective_str(str(nodeid),model,params)
            parametrizer = params['deepviz_param']
            optimizer = params['deepviz_optim']
            transforms = params['deepviz_transforms']
            image_size = params['deepviz_image_size']
            deepviz_neuron = params['deepviz_neuron']
            param_str = object_2_str(parametrizer,"params['deepviz_param']=")
            optimizer_str = object_2_str(optimizer,"params['deepviz_optim']=")
            transforms_str = object_2_str(transforms,"params['deepviz_transforms']=")
            df_sel = images_df.loc[(images_df['targetid'] == str(nodeid)) & (images_df['objective'] == objective_str) & (images_df['parametrizer'] == param_str) & (images_df['optimizer'] == optimizer_str) & (images_df['transforms'] == transforms_str) & (images_df['neuron'] == deepviz_neuron)]
            image_name = df_sel.iloc[0]['image_name']
            print('found image %s'%image_name)
            image_path = params['prepped_model_path']+'/visualizations/images/%s/%s'%(type_names[deepviz_neuron],image_name)
            image = preprocess_image(image_path,params)
            output = model_dis(image)
            layer_activations = get_activations_from_dissected_Conv2d_modules(model_dis)
            target_pixels = np.unravel_index(np.argmax(layer_activations['nodes'][layer_num][0][within_layer_nodeid], axis=None), layer_activations['nodes'][layer_num][0][within_layer_nodeid].shape)
            for other_within_layer_nodeid, other_nodeid in enumerate(params['layer_nodes'][layer_num][1]):
                activations[-1][int(other_within_layer_nodeid)][within_layer_nodeid] = layer_activations['nodes'][layer_num][0][other_within_layer_nodeid][target_pixels[0]][target_pixels[1]]           
    return activations

print('Getting model activations from deepviz images')

start = time.time()

activations = gen_deepviz_activations(model_dis,params)
pickle.dump(activations,open(params['prepped_model_path']+'/deepviz_activations.pkl','wb')) 


print('DEEPVIZ ACTIVATIONS TIME: %s'%str(time.time()-start))