#various functions for generating subgraphs
import os
from copy import deepcopy
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objs as go
from visualizer_scripts.visualizer_helper_functions import *
import sys
#sys.path.insert(0, os.path.abspath('../prep_model_scripts/'))
from prep_model_scripts.dissected_Conv2d import *
from prep_model_scripts.data_loading_functions import *



def filter_edges_by_nodes(edges_df,thresholded_nodes_df):
	valid_nodes = {}

	for row in thresholded_nodes_df.itertuples():
		if row.layer not in valid_nodes.keys():
			valid_nodes[row.layer] = [row.node_num_by_layer]
		else:
			valid_nodes[row.layer].append(row.node_num_by_layer)
			
	# def filter_edges_fn(row):
	# 	try: 
	# 		if (row['out_channel'] in valid_nodes[row['layer']]): #try block because maybe no valid nodes in row['layer']
	# 			if row['layer'] == 0:
	# 				return True
	# 			elif row['in_channel'] in valid_nodes[row['layer']-1]:
	# 				return True
	# 	except:
	# 		return False
	# 	return False
	
	# mask = edges_df.apply(filter_edges_fn, axis=1)
	# return edges_df[mask]
	filtered_df = pd.DataFrame(columns = edges_df.columns)
	entries = []
	for i in valid_nodes:
		entry = edges_df.loc[(edges_df['out_channel'].isin(valid_nodes[i])) & (edges_df['layer'] == i)]
		if i != 0:
			entry =  entry.loc[entry['in_channel'].isin(valid_nodes[i-1])]
		entries.append(entry)

	found_df = pd.concat(entries)
	filtered_df = pd.concat([filtered_df, found_df])
	return filtered_df


def hierarchically_threshold_edges(threshold,rank_type,edges_df,nodes_thresholded_df):          #just get those edges that pass the threshold criteria for the target category
	if len(threshold) != 2:
		raise Exception('length of threshold needs to be two ([lower, higher])')
		
	valid_nodes = {}
	for row in nodes_thresholded_df.itertuples():
		if row.layer not in valid_nodes.keys():
			valid_nodes[row.layer] = [row.node_num_by_layer]
		else:
			valid_nodes[row.layer].append(row.node_num_by_layer)

	
	filtered_df = pd.DataFrame(columns = edges_df.columns)
	
	layers_edges = []
	for layer in valid_nodes:
		layer_edges = edges_df.loc[(edges_df['out_channel'].isin(valid_nodes[layer])) & (edges_df['layer'] == layer)]
		if layer != 0:
			layer_edges =  layer_edges.loc[layer_edges['in_channel'].isin(valid_nodes[layer-1])]
			
		nodes_edges = []
		for node in valid_nodes[layer]:
			node_edges = layer_edges.loc[layer_edges['out_channel']== node]
			minmax = [node_edges[rank_type+'_rank'].min(),node_edges[rank_type+'_rank'].max()]
			minmax_t = [threshold[0]*(minmax[1]-minmax[0])+minmax[0],threshold[1]*(minmax[1]-minmax[0])+minmax[0]]
			node_edges = node_edges.loc[(node_edges[rank_type+'_rank'] >= minmax_t[0]) & (node_edges[rank_type+'_rank'] <= minmax_t[1])]
			nodes_edges.append(node_edges)
										 
		layer_edges= pd.concat(nodes_edges)                               
		layers_edges.append(layer_edges)

	found_df = pd.concat(layers_edges)
	filtered_df = pd.concat([filtered_df, found_df])     
	
	return filtered_df

def hierarchical_accum_threshold(threshold_node,threshold_edge,rank_type,edges_df,nodes_df,ascending=False):
	threshed_nodes_df = get_accum_thresholded_ranksdf(threshold_node,rank_type,nodes_df, ascending=ascending)
	
	valid_nodes = {}
	for row in threshed_nodes_df.itertuples():
		if row.layer not in valid_nodes.keys():
			valid_nodes[row.layer] = [row.node_num_by_layer]
		else:
			valid_nodes[row.layer].append(row.node_num_by_layer)


	threshed_edges_df = pd.DataFrame(columns = edges_df.columns)
	
	layers_edges = []
	for layer in valid_nodes:
		if isinstance(threshold_edge,float):
			thresh_edge = threshold_edge
		else:
			thresh_edge = threshold_edge[layer]
		layer_edges = edges_df.loc[(edges_df['out_channel'].isin(valid_nodes[layer])) & (edges_df['layer'] == layer)]
		if layer != 0:
			layer_edges =  layer_edges.loc[layer_edges['in_channel'].isin(valid_nodes[layer-1])]

		nodes_edges = []
		for node_out in layer_edges['out_channel'].unique():
			node_out_edges = layer_edges.loc[layer_edges['out_channel']== node_out]
			node_out_edges = node_out_edges.sort_values(rank_type+'_rank',ascending=ascending)
			total_imp = node_out_edges[rank_type+'_rank'].sum()
			running_total = 0
			for i in range(len(node_out_edges)):
				running_total+=node_out_edges.iloc[i][rank_type+'_rank']
				if running_total/total_imp >= thresh_edge:
					break
			node_out_edges = node_out_edges.iloc[0:i+1]
			nodes_edges.append(node_out_edges)
		for node_in in layer_edges['in_channel'].unique():
			node_in_edges = layer_edges.loc[layer_edges['in_channel']== node_in]
			node_in_edges = node_in_edges.sort_values(rank_type+'_rank',ascending=ascending)
			total_imp = node_in_edges[rank_type+'_rank'].sum()
			running_total = 0
			for i in range(len(node_in_edges)):
				running_total+=node_in_edges.iloc[i][rank_type+'_rank']
				if running_total/total_imp >= thresh_edge:
					break
			node_in_edges = node_in_edges.iloc[0:i+1]
			nodes_edges.append(node_in_edges)
			
		layer_edges= pd.concat(nodes_edges).drop_duplicates()                               
		layers_edges.append(layer_edges)

	found_df = pd.concat(layers_edges).drop_duplicates()
	threshed_edges_df = pd.concat([threshed_edges_df, found_df])     
	
	return threshed_nodes_df,threshed_edges_df


def get_accum_thresholded_ranksdf(threshold,rank_type,df, ascending=False):          #just get those edges that pass the threshold criteria for the target category
	layers = df['layer'].unique()
	print(layers)
	layers.sort()
	threshold_df = pd.DataFrame(columns = df.columns)
	
	layers_dfs = []   
	for layer in layers:
		if isinstance(threshold,float):
			thresh = threshold
		else:
			thresh = threshold[layer]
		layer_df = df.loc[df['layer'] == layer]
		if not (layer_df[rank_type+'_rank'].max()> 0):
			continue
		layer_df = layer_df.sort_values(rank_type+'_rank',ascending=ascending)
		
		total_imp = layer_df[rank_type+'_rank'].sum()
		

		running_total = 0
		for i in range(len(layer_df)):
			running_total+=layer_df.iloc[i][rank_type+'_rank']
			if running_total/total_imp >= thresh:
				break
		layer_df = layer_df.iloc[0:i+1]

		layers_dfs.append(layer_df)
	
	found_df = pd.concat(layers_dfs)
	threshold_df = pd.concat([threshold_df, found_df])
	return threshold_df

def hierarchical_size_threshold(node_size,edge_size,rank_type,nodes_df,edges_df,selection='best'):
	"""This function currently requires list as 'size', (one size per layer) """
	threshed_nodes_df = get_size_thresholded_ranksdf(node_size,rank_type,nodes_df, selection=selection)
	
	valid_nodes = {}
	for row in threshed_nodes_df.itertuples():
		if row.layer not in valid_nodes.keys():
			valid_nodes[row.layer] = [row.node_num_by_layer]
		else:
			valid_nodes[row.layer].append(row.node_num_by_layer)


	threshed_edges_df = pd.DataFrame(columns = edges_df.columns)
	layers_edges = []
	for layer in valid_nodes:
		if isinstance(edge_size,int):
			tot_edge_num = edge_size
		else:
			try:
				tot_edge_num = edge_size[layer]
			except:
				continue

		layer_edges = edges_df.loc[(edges_df['out_channel'].isin(valid_nodes[layer])) & (edges_df['layer'] == layer)]
		if layer != 0:
			layer_edges =  layer_edges.loc[layer_edges['in_channel'].isin(valid_nodes[layer-1])]
		
		nodes_edges = []
		#first get at least one output and input edge per node
		for node_out in layer_edges['out_channel'].unique():
			node_out_edges = layer_edges.loc[layer_edges['out_channel']== node_out]
			if selection == 'best':
				node_out_edges = node_out_edges.sort_values(rank_type+'_rank',ascending=False)
			elif selection == 'worst':
				node_out_edges = node_out_edges.sort_values(rank_type+'_rank',ascending=True)
			else:
				node_out_edges = node_out_edges.sample(frac=1)
			#import pdb;pdb.set_trace()
			node_out_edges = node_out_edges.iloc[[0]]
			nodes_edges.append(node_out_edges)
		for node_in in layer_edges['in_channel'].unique():
			node_in_edges = layer_edges.loc[layer_edges['in_channel']== node_in]
			if selection == 'best':
				node_in_edges = node_in_edges.sort_values(rank_type+'_rank',ascending=False)
			elif selection == 'worst':
				node_in_edges = node_in_edges.sort_values(rank_type+'_rank',ascending=True)
			else:
				node_in_edges = node_in_edges.sample(frac=1)
			node_in_edges = node_in_edges.iloc[[0]]
			nodes_edges.append(node_in_edges)
		#import pdb;pdb.set_trace()	
		select_layer_edges= pd.concat(nodes_edges).drop_duplicates()
		edge_num = len(select_layer_edges) 
		nonselect_layer_edges = layer_edges.loc[~layer_edges.index.isin(list(select_layer_edges.index))] 
		remaining_edges = tot_edge_num - edge_num
		if selection == 'best':
			nonselect_layer_edges = nonselect_layer_edges.sort_values(rank_type+'_rank',ascending=False)
		elif selection == 'worst':
			nonselect_layer_edges = nonselect_layer_edges.sort_values(rank_type+'_rank',ascending=True)
		else:
			nonselect_layer_edges = nonselect_layer_edges.sample(frac=1)
		additional_edges = nonselect_layer_edges.iloc[0:remaining_edges]
		found_layer_edges = pd.concat([select_layer_edges,additional_edges]).drop_duplicates()

		layers_edges.append(found_layer_edges)

	found_df = pd.concat(layers_edges).drop_duplicates()
	threshed_edges_df = pd.concat([threshed_edges_df, found_df])     
	# threshed_edges_df = threshed_edges_df.astype({'edge_num': 'int64',
	# 											  'layer': 'int64',
	# 											  'out_channel': 'int64',
	# 											  'in_channel': 'int64'
	# 											  })
	return threshed_nodes_df,threshed_edges_df

def get_size_thresholded_ranksdf(size,rank_type,df, selection='best'):
	"""This function currently requires list as 'size', (one size per layer) """

	layers = df['layer'].unique()
	#print(layers)
	layers.sort()
	threshold_df = pd.DataFrame(columns = df.columns)
	
	layers_dfs = []   
	for layer in layers:
		if isinstance(size,int):
			tot_num = size
		else:
			try:
				tot_num = size[layer]
			except:
				continue
		layer_df = df.loc[df['layer'] == layer]
		if not (layer_df[rank_type+'_rank'].max()> 0):
			continue
		if selection == 'best':
			layer_df = layer_df.sort_values(rank_type+'_rank',ascending=False)
		elif selection == 'worst':
			layer_df = layer_df.sort_values(rank_type+'_rank',ascending=True)
		else:
			layer_df = layer_df.sample(frac=1)

		layer_df = layer_df.iloc[0:tot_num]

		layers_dfs.append(layer_df)
	#fix layers_dfs so last layer node is target node no matter what
	layer = int(layers_dfs[-1].iloc[[0]]['layer'])
	layers_dfs.pop()
	layer_df = df.loc[df['layer'] == layer]
	layer_df = layer_df.sort_values(rank_type+'_rank',ascending=False)
	layer_df = layer_df.iloc[0:tot_num]
	layers_dfs.append(layer_df)
	
	found_df = pd.concat(layers_dfs)
	threshold_df = pd.concat([threshold_df, found_df])
	# threshold_df = threshold_df.astype({'node_num': 'int64',
	# 									'layer': 'int64',
	# 									'node_num_by_layer': 'int64'
	# 									})
	return threshold_df

def extract_subgraph(model,thresholded_nodes_df,thresholded_egdes_df,params, save=True):
	#this is currently hacky only works on models with all nn.sequential or .features module
	model.to('cpu')
	l = 0
	subgraph_model = nn.Sequential()
	
	#hack
	for layer in model.children():
		if not isinstance(layer, nn.Conv2d):
			model = model.features
			break
		break
	 
	with torch.no_grad():
		for layer in model.children():
			if isinstance(layer, nn.Conv2d):
				name = 'conv_{}'.format(l)
			elif isinstance(layer, nn.ReLU):
				name = 'relu_{}'.format(l)
				layer = nn.ReLU(inplace=False)
			elif isinstance(layer, nn.MaxPool2d):
				name = 'pool_{}'.format(l)
			elif isinstance(layer, nn.BatchNorm2d):
				name = 'bn_{}'.format(l)
			else:
				raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
			if not isinstance(layer, nn.Conv2d):
				subgraph_model.add_module(name, layer)
			else:
				print('layer: %s'%str(l))
				old_conv = layer
				out_channels = list(thresholded_nodes_df.loc[thresholded_nodes_df['layer'] == l]['node_num_by_layer'].sort_values().unique())
				num_out_channels = len(out_channels)
				if l == 0:
					num_in_channels = old_conv.in_channels
					in_channels = list(range(num_in_channels))
				else:
					in_channels = list(thresholded_nodes_df.loc[thresholded_nodes_df['layer'] == l-1]['node_num_by_layer'].sort_values().unique())
					num_in_channels = len(in_channels)
				new_conv = \
						torch.nn.Conv2d(in_channels = num_in_channels, \
						out_channels = num_out_channels ,
						kernel_size = old_conv.kernel_size, \
						stride = old_conv.stride,
						padding = old_conv.padding,
						dilation = old_conv.dilation,
						groups = old_conv.groups,
						bias = (old_conv.bias is not None))                
			
				#full_weights = []
				#GENERATE WEIGHT MATRIX
				weights = new_conv.weight
				weights.fill_(0.)
				#print(out_channels)
				#print(len(out_channels))
				for o_i,o in enumerate(out_channels):
					for i_i,row in enumerate(thresholded_egdes_df.loc[(thresholded_egdes_df['layer'] == l) & (thresholded_egdes_df['out_channel'] == o)].sort_values('in_channel').itertuples()):
						weights[o_i,in_channels.index(row.in_channel),:,:] = old_conv.weight[o,row.in_channel,:,:]
				#for node in thresholded_nodes_df.loc[thresholded_nodes_df['layer'] == l].sort_values('node_num_by_layer')['node_num_by_layer']:
					#print(node)
					#ws.append(weights[node,:,:,:].unsqueeze(0))
				
				#GENERATE BIAS 
				if new_conv.bias is not None:
					for o_i,o in enumerate(out_channels):
						new_conv.bias[o_i] = old_conv.bias[o]
				
				
				subgraph_model.add_module(name, new_conv)
				#next layer
				l += 1

			if l not in thresholded_nodes_df['layer'].unique():
				break
    
				
	return subgraph_model





def get_layer_sizes_from_df(df):
    '''takes a df with a "layer" column and returns a list of layerwise sizes '''
    layers = df['layer'].unique()
    layers.sort()
    output = []
    for layer in layers:
        output.append(len(df.loc[df['layer']==layer]))
    return output


def make_same_sized_subgraph(sub_dict_path,model,selection='random',node=None,save=False):
    #import pdb;pdb.set_trace()
    sub_dict = torch.load(sub_dict_path)
    if 'input' in sub_dict['gen_params'].keys():
        inputs = sub_dict['gen_params']['input']
    else:
        inputs = 'small_SPAN'
    if 'output' in sub_dict['gen_params'].keys():
        output = int(sub_dict['gen_params']['output'])
    elif node is None:
        int(sub_dict_path.split('/')[-1].split('_')[0])
    else:
        output = int(node)
    n_df,e_df=ranksdf_store(inputs, output, [], model_dis=model_dis)
    n_df = minmax_normalize_ranks_df(n_df,params)
    n_sizes = get_layer_sizes_from_df(sub_dict['node_df'])
    e_sizes = get_layer_sizes_from_df(sub_dict['edge_df'])
    size_n_df,size_e_df = hierarchical_size_threshold(n_sizes,e_sizes,rank_type,n_df,e_df,selection=selection)
    size_model = extract_subgraph(model,size_n_df,size_e_df,params)
    save_object = {
                   'model':size_model,
                   'gen_params':
                        {'node_sizes':n_sizes,
                         'edge_sizes':e_sizes,
                         'input':inputs,
                         'output':output,
                         'selection':selection,
                         'from_model':sub_dict_path
                        },
                    'node_df':size_n_df,
                    'edge_df':size_e_df
                  }
    if save:
        print('saving model to %s'%save)
        torch.save(save_object,save)
    return save_object
        

def same_node_subgraph_responses(node,model,submodel, category, params,batch_size):
    '''model is model from which subgraph was extracted, terminating in "node" (use unique node id)'''
    device = params['device']
    node_layer,node_within_layer_id,node_layer_name = nodeid_2_perlayerid(node,params)
    model.to(device)
    submodel.to(device)
    
    ###Hook model
    model_activations = []
    def get_activation(node_within_layer_id=node_within_layer_id):
        def hook(model, input, output):
            model_activations.append(output[:,node_within_layer_id,:,:].detach())
            #if model_activations is None:
            #    model_activations = output[:,node,:,:].detach()
            #else:
            #    model_activations = torch.cat((model_activations,output[:,node,:,:].detach()), 0)
        return hook
    
    ##### THIS IS NOT MODEL GENERAL, ASSUMING 'FEATURES', basically alexnet only right now
    handle = model.features[int(node_layer_name.split('_')[-1])].register_forward_hook(get_activation())
    
    submodel_activations = None
    
    ####SET UP DATALOADER
    kwargs = {'num_workers': params['num_workers'], 'pin_memory': True} if ('cuda' in params['device']) else {}
    
    if category =='overall':
        categories = os.listdir(params['rank_img_path'])
    else:
        categories = [category]
    for cat in categories:

        image_loader = torch.utils.data.DataLoader(
                rank_image_data(params['rank_img_path']+'/'+cat,params['preprocess'],params['label_file_path']),
                batch_size=batch_size,
                shuffle=True,
                **kwargs)
        ###run images through
        for i, (batch, target) in enumerate(image_loader):
            #print('batch %s'%i)
            batch, target = batch.to(device), target.to(device)
            model_output = model(batch)    #running forward pass sets up hooks and stores activations in each dissected_Conv2d module
            submodel_output = submodel(batch)
            if submodel_activations is None:
                submodel_activations = submodel_output.detach()
            else:
                submodel_activations = torch.cat((submodel_activations,submodel_output.detach()), 0)
    relu = nn.ReLU(inplace=True)          
    model_activations = relu(torch.cat(model_activations, 0))
    submodel_activations = relu(torch.squeeze(submodel_activations,1))
    #print(model_activations.shape)
    #print(submodel_activations.shape)
    handle.remove()
    return torch.flatten(model_activations),torch.flatten(submodel_activations)
            


        