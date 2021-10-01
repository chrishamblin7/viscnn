#functions for ablating model
import os
from copy import deepcopy
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objs as go
from viscnn.ablations import *
import sys
sys.path.insert(0, os.path.abspath('../prep_model_scripts/'))
from viscnn.dissected_Conv2d import *
from viscnn.data_loading import *


def ablate_model(target,model, params, put_back = False):
	edge_target = False
	if '-' in target:
		valid, from_layer,to_layer,from_within_id,to_within_id = check_edge_validity(target,params)
		if not valid:
			raise ValueError('edgename %s is invalid'%edgename)
		conv_module = layer_2_dissected_conv2d(int(to_layer),model)[0]
		preadd_id = conv_module.add_indices[int(to_within_id)][int(from_within_id)]
		if put_back:
			conv_module.edge_ablations[:] = [x for x in conv_module.edge_ablations if x != int(preadd_id)]
		elif int(preadd_id) not in conv_module.edge_ablations:
			conv_module.edge_ablations.append(int(preadd_id))   
	else:
		layer,within_layer_id,layer_name = nodeid_2_perlayerid(target,params)
		conv_module = layer_2_dissected_conv2d(int(layer),model)[0]
		if put_back:
			conv_module.node_ablations[:] = [x for x in conv_module.node_ablations if x != int(within_layer_id)]
		elif int(within_layer_id) not in conv_module.node_ablations:
			conv_module.node_ablations.append(int(within_layer_id))   
		
def ablation_text_2_list(text, params):
	#turn to list
	l = text.split(',')
	l = [x.strip() for x in l]
	l = list(set(l))
	l.sort()
	output_list = []
	for x in l:
		if x.isnumeric():
			if int(x) >= params['num_nodes']:
				print('"%s" in ablation list is not a node'%x)
				continue
			output_list.append(x)
		elif '-' in x:
			valid = check_edge_validity(x,params)[0]
			if valid:
			   output_list.append(x)
			else:
				print('"%s" in ablation list is not an edge'%x) 
		else:
			print('"%s" in ablation list is not a node/edge'%x)
	return output_list

def ablate_model_with_list(ablation_list,model,params):
	model = set_across_model(model,'clear_ablations',None)
	for target in ablation_list:
		ablate_model(target,model,params)
