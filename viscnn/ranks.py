#functions related to formatting 'ranks',
# the pruning scores associated with network graph components

import os
from copy import deepcopy
import pandas as pd
import numpy as np
from viscnn.dissected_Conv2d import *
from viscnn.data_loading import *
from viscnn.utils import *








def get_model_ranks_for_category(category, target_node, model_dis,params):
	print('running model to get ranks for "%s" on target "%s"'%(str(category),str(target_node)))
	device = params['device']

	print('using device %s'%device)
	criterion = params['criterion']
	####SET UP MODEL
	model_dis = set_across_model(model_dis,'target_node',None)
	if target_node != 'loss':
		target_node_layer,target_node_within_layer_id,target_node_layer_name = nodeid_2_perlayerid(target_node,params)
		model_dis=set_model_target_node(model_dis,target_node_layer,target_node_within_layer_id)

	model_dis = set_across_model(model_dis,'clear_ranks',False)
	model_dis.to(device)
	node_ranks = {}
	edge_ranks = {}


	####SET UP DATALOADER
	kwargs = {'num_workers': params['num_workers'], 'pin_memory': True} if ('cuda' in params['device']) else {}

	if category =='overall':
		categories = os.listdir(params['rank_img_path'])
	else:
		categories = [category]
	for cat in categories:

		image_loader = torch.utils.data.DataLoader(
				rank_image_data(params['rank_img_path']+'/'+cat,params['preprocess'],params['label_file_path']),
				batch_size=params['batch_size'],
				shuffle=True,
				**kwargs)	

		##RUNNING DATA THROUGH MODEL
		#Pass data through model in batches
		for i, (batch, target) in enumerate(image_loader):
			print('batch %s'%i)
			model_dis.zero_grad()
			batch, target = batch.to(device), target.to(device)
			try:
				output = model_dis(batch)    #running forward pass sets up hooks and stores activations in each dissected_Conv2d module
				if target_node == 'loss':
					target = max_likelihood_for_no_target(target,output) 
					criterion(output, target).backward()    #running backward pass calls all the hooks and calculates the ranks of all edges and nodes in the graph 
			except TargetReached:
				print('target node %s reached, halted forward pass'%str(target_node))

			#	torch.sum(output).backward()    # run backward pass with respect to net outputs rather than loss function

	layer_ranks = get_ranks_from_dissected_Conv2d_modules(model_dis)
	model_dis = clear_ranks_across_model(model_dis)
	model_dis = set_across_model(model_dis,'clear_ranks',True)

	return layer_ranks

def get_model_ranks_from_image(image_path, target_node, model_dis, params): 
	print('running model to get ranks for image: %s'%str(image_path))
	#model_dis.clear_ranks_func()  #so ranks dont accumulate

	device = params['device']

	criterion = params['criterion']
	#image loading 
	image_name = image_path.split('/')[-1]
	image,target = single_image_loader(image_path, params['preprocess'], label_file_path = params['label_file_path'])
	image, target = image.to(device), target.to(device)

	model_dis = set_across_model(model_dis,'target_node',None)
	if target_node != 'loss':
		target_node_layer,target_node_within_layer_id,target_node_layer_name = nodeid_2_perlayerid(target_node,params)
		model_dis=set_model_target_node(model_dis,target_node_layer,target_node_within_layer_id)
	model_dis.to(device)

	#pass image through model
	try:
		output = model_dis(image)    #running forward pass sets up hooks and stores activations in each dissected_Conv2d module
		if target_node == 'loss':
			target = max_likelihood_for_no_target(target,output) 
			criterion(output, target).backward()    #running backward pass calls all the hooks and calculates the ranks of all edges and nodes in the graph 
	except TargetReached:
		print('target node %s reached, halted forward pass'%str(target_node))

	layer_ranks = get_ranks_from_dissected_Conv2d_modules(model_dis)
	return layer_ranks



def rank_file_2_df(file_path):      
	'''
	takes a node or edge 'rank.pt' file and turns it into a pandas dataframe, 
	or takes the dict itself not file path
	'''
	file_name = file_path.split('/')[-1]
	category = file_name.replace('_edges_rank.pt','').replace('_nodes_rank.pt','')
	ranks = torch.load(file_path)
	rank_types = list(ranks.keys())

	if 'weight' in rank_types:
		node_column_names = ['node_num','layer_name','layer','node_num_by_layer','weight_rank']
		edge_column_names = ['edge_num','layer_name','layer','out_channel','in_channel','weight_rank']
	else:
		node_column_names = ['node_num','layer_name','layer','node_num_by_layer','act_rank','grad_rank','actxgrad_rank']
		edge_column_names = ['edge_num','layer_name','layer','out_channel','in_channel','act_rank','grad_rank','actxgrad_rank']

	#nodes
	if 'node' in file_path.split('/')[-1]:
		node_dflist = []
		node_num = 0
		for layer in range(len(ranks[rank_types[0]])):
			layer_name = ranks[rank_types[0]][layer][0]
			for num_by_layer in range(len(ranks[rank_types[0]][layer][1])):
				if 'weight' in rank_types:
					node_dflist.append([node_num,layer_name,layer,num_by_layer,ranks['weight'][layer][1][num_by_layer]])              
				else:
					node_dflist.append([node_num,layer_name,layer,num_by_layer,ranks['act'][layer][1][num_by_layer],
										ranks['grad'][layer][1][num_by_layer],ranks['actxgrad'][layer][1][num_by_layer]])
				node_num += 1
		#make nodes DF
		df = pd.DataFrame(node_dflist,columns=node_column_names)

	elif 'edge' in file_path.split('/')[-1]:
		#edges
		edge_dflist = []
		edge_num = 0
		for layer in range(len(ranks[rank_types[0]])):
			layer_name = ranks[rank_types[0]][layer][0]
			for out_channel in range(len(ranks[rank_types[0]][layer][1])):
				for in_channel in range(len(ranks[rank_types[0]][layer][1][out_channel])):
					if 'weight' in rank_types:
						edge_dflist.append([edge_num,layer_name,layer,out_channel,in_channel,ranks['weight'][layer][1][out_channel][in_channel]])
					else:
						edge_dflist.append([edge_num,layer_name,layer,out_channel,in_channel,ranks['act'][layer][1][out_channel][in_channel],
											ranks['grad'][layer][1][out_channel][in_channel],ranks['actxgrad'][layer][1][out_channel][in_channel]])
					edge_num += 1
		df = pd.DataFrame(edge_dflist,columns=edge_column_names)
	
	else:
		raise Exception("Can't determine if %s is node or edge rank. Make sure 'node' or 'edge' is in file name"%file_path)

	return df



def rank_dict_2_df(ranks):      
	'''
	takes a rank dictionary and turns it into a pandas dataframe
	'''
	rank_types = list(ranks['nodes'].keys())
	node_column_names = ['node_num','layer_name','layer','node_num_by_layer','act_rank','grad_rank','actxgrad_rank']
	edge_column_names = ['edge_num','layer_name','layer','out_channel','in_channel','act_rank','grad_rank','actxgrad_rank']

	#nodes

	node_dflist = []
	node_num = 0
	for layer in range(len(ranks['nodes'][rank_types[0]])):
		layer_name = ranks['nodes'][rank_types[0]][layer][0]
		for num_by_layer in range(len(ranks['nodes'][rank_types[0]][layer][1])):
			node_dflist.append([node_num,layer_name,layer,num_by_layer,ranks['nodes']['act'][layer][1][num_by_layer],
								ranks['nodes']['grad'][layer][1][num_by_layer],ranks['nodes']['actxgrad'][layer][1][num_by_layer]])
			node_num += 1
	#make nodes DF
	nodes_df = pd.DataFrame(node_dflist,columns=node_column_names)


	#edges
	edge_dflist = []
	edge_num = 0
	for layer in range(len(ranks['edges'][rank_types[0]])):
		layer_name = ranks['edges'][rank_types[0]][layer][0]
		for out_channel in range(len(ranks['edges'][rank_types[0]][layer][1])):
			for in_channel in range(len(ranks['edges'][rank_types[0]][layer][1][out_channel])):
				edge_dflist.append([edge_num,layer_name,layer,out_channel,in_channel,ranks['edges']['act'][layer][1][out_channel][in_channel],
									ranks['edges']['grad'][layer][1][out_channel][in_channel],ranks['edges']['actxgrad'][layer][1][out_channel][in_channel]])
				edge_num += 1
	edges_df = pd.DataFrame(edge_dflist,columns=edge_column_names)

	return nodes_df, edges_df
	



def minmax_normalize_ranks_df(df,params,weight=False):
	if weight:
		rank_types = ['weight']
	else:
		rank_types = ['act','grad','actxgrad']

	for rank_type in rank_types:
		for layer in range(params['num_layers']):
			col = df.loc[df['layer']==layer][rank_type+'_rank']
			maximum = np.max(col)
			minimum = np.min(col)
			if maximum == 0:
				print('maximum value 0 for rank type %s and layer %s'%(rank_type,str(layer)))
			else:
				df[rank_type+'_rank'] = np.where(df['layer'] == layer ,(df[rank_type+'_rank']-minimum)/(maximum-minimum),df[rank_type+'_rank'] )
	
	return df



def get_thresholded_ranksdf(threshold,rank_type,df):          #just get those edges that pass the threshold criteria for the target category
	if len(threshold) != 2:
		raise Exception('length of threshold needs to be two ([lower, higher])')
	return df.loc[(df[rank_type+'_rank'] >= threshold[0]) & (df[rank_type+'_rank'] <= threshold[1])]

