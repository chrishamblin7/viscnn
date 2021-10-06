#various functions used by the visualizer tool
import os
from copy import deepcopy
import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import sys
#sys.path.insert(0, os.path.abspath('../prep_model_scripts/'))
from viscnn.dissected_Conv2d import *
from viscnn.data_loading import *
from viscnn.ranks import *
from viscnn.activations import *
from viscnn.contrasts import *
from viscnn.featureviz import *
from viscnn.ablations import *
from viscnn.subgraphs import *
from viscnn.utils import *
from viscnn.visualizers.layouts import *


import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
#from dash.exceptions import PreventUpdate
#import utils.dash_reusable_components as drc
import flask
import os
import json
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from flask_caching import Cache



  
### HELPER FUNCTIONS

#INPUT IMAGE FUNCTIONS

def image2plot(image_path,layout,resize = False,size = (32,32)):
	if isinstance(image_path,str):
		img = Image.open(image_path)
	else:
		img=image_path
	if resize:
		img = img.resize(size,resample=Image.NEAREST)
	
	#fig = go.Figure(data=go.Image(z=img,dx=244,dy=244))
	trace1 = go.Scatter(x=[],y=[])
	fig=go.Figure(data=[trace1],layout=layout)
	fig.update_layout(images= [dict(
									source= img,
									xref= "x",
									yref= "y",
									x= 0,
									y= 10,
									sizex= 10,
									sizey= 10,
									sizing= "stretch",
									opacity= 1,
									layer= "below"
									)
								]
					 )
	return fig

def image2heatmap(image_path,layout,resize = False,size = (32,32)):          #displays image as a plotly heatmap object, with colors preserved
	
	img = Image.open(image_path)
	if resize:
		img = img.resize(size,resample=Image.NEAREST)
	np_img = np.array(img)    
	
	if len(np_img.shape) == 2: #grayscale img
		colorscale = [[0,"black"], [1,"white"]]
		heatmap = go.Heatmap(z=np.flip(np.array(np_img),0), 
				 colorscale = colorscale, 
				 showscale = False) 
		
	else:   #rgb image
		pixels = img.convert('RGBA').load() #rgba values
		width, height = img.size   #width and height of image
		num_pixels = width*height
		step = 1/num_pixels

		colorscale = []           
		z = []

		i = 0
		for y in range(height):
			z.append([])
			for x in range(width):
				z[-1].append(step*(i+.5))
				r, g, b, a = pixels[x, y]
				colorscale.append([step*i,rgb2hex(r, g, b)])
				colorscale.append([step*(i+1),rgb2hex(r, g, b)])
				i+=1     
		heatmap = go.Heatmap(z=np.flip(np.array(z),0), 
						 colorscale = colorscale, 
						 showscale = False)
		
		
	fig = go.Figure(data=[heatmap],layout=layout)

	# fig.update_layout(width=350, 
	#                   height=350,
	#                   uirevision = True,
	#                   margin=dict(
	#                     l=1,
	#                     r=1,
	#                     b=1,
	#                     t=1,
	#                     pad=1)
	#                 )
	return fig




#NODE FUNCTIONS

def node_color_scaling(x,node_min,min_color=.4):
	if node_min is None:
		return x
	else:
		if x>=node_min:
			return (1-min_color)*(x-node_min)/(1-node_min)+min_color
		else:
			return 0
	#return -(x-1)**4+1

def gen_node_colors(nodes_df,rank_type,params,node_min=None):
	layer_nodes = params['layer_nodes']
	layer_colors = params['layer_colors']
	node_colors = []
	node_weights = []
	for layer in range(len(layer_nodes)):
		node_colors.append([])
		node_weights.append([])
		for node in layer_nodes[layer][1]:
			node_weight = nodes_df.iloc[node][rank_type+'_rank']
			node_weights[-1].append(node_weight)
			if node_min is None:
				alpha = node_color_scaling(node_weight,node_min)
			else:
				alpha = node_color_scaling(node_weight,node_min[layer])
			node_colors[-1].append(layer_colors[layer%len(layer_colors)]+str(round(alpha,3))+')')
			
	return node_colors,node_weights



#EDGE FUNCTIONS



def edge_width_scaling(x):
	#return max(.4,(x*10)**1.7)
	return max(.4,np.exp(2.5*x))

def edge_color_scaling(x):
	#return max(.7,-(x-1)**4+1)
	return max(.7, x)


def get_max_edge_widths(edge_widths):
	maxes = []
	for layer in range(len(edge_widths)):
		if len(edge_widths[layer]) >0:
			maxes.append(edge_widths[layer].index(max(edge_widths[layer])))
		else:
			maxes.append(None)
	return maxes

def gen_edge_graphdata(df, node_positions, rank_type, target_category, params,kernel_colors, num_hoverpoints=15):
	layer_nodes = params['layer_nodes']
	layer_colors = params['layer_colors']
	imgnode_positions = params['imgnode_positions']
	imgnode_names = params['imgnode_names']
	num_layers = params['num_layers']

	edges_df_columns = list(df.columns)
	edge_positions = []
	colors = []
	widths = []
	weights = []
	names = []
	#max_weight = 0
	for row in df.itertuples():
		while row.layer > len(edge_positions): # we skipped a layer, its got no edges in threshold, so lets add empty lists
			edge_positions.append({'X':[],'Y':[],'Z':[]})
			colors.append([])
			widths.append([])
			weights.append([])
			names.append([])  
		if row.layer == len(edge_positions):
			edge_positions.append({'X':[],'Y':[],'Z':[]})
			colors.append([])
			widths.append([])
			weights.append([])
			names.append([])        
		#position
		for dim in ['X','Y','Z']:
			end_pos = node_positions[row.layer][dim][row.out_channel]
			if row.layer != 0:
				start_pos = node_positions[row.layer-1][dim][row.in_channel]
			else:
				start_pos = imgnode_positions[dim][row.in_channel]
			
			step = (end_pos-start_pos)/(num_hoverpoints+1)
			points = [start_pos]
			for i in range(1,num_hoverpoints+1):
				points.append(start_pos+i*step)
			points.append(end_pos)
			edge_positions[row.layer][dim].append(points)
		#width
		widths[row.layer].append(edge_width_scaling(row[edges_df_columns.index(rank_type+'_rank')+1]))
		#weight
		weights[row.layer].append(row[edges_df_columns.index(rank_type+'_rank')+1])
		#max_weight = max(max_weight, row.rank_score)
		#names
		out_node = layer_nodes[row.layer][1][row.out_channel]
		if row.layer != 0:
			in_node = layer_nodes[row.layer-1][1][row.in_channel]
		else:
			in_node = imgnode_names[row.in_channel]
		names[row.layer].append(str(in_node)+'-'+str(out_node))
		#color
		if kernel_colors is None:
			alpha = edge_color_scaling(row[edges_df_columns.index(rank_type+'_rank')+1])
			colors[row.layer].append(layer_colors[row.layer%len(layer_colors)]+str(round(alpha,3))+')')
		else:
			colors[row.layer].append(color_vec_2_str(kernel_colors[int(row.layer)][int(row.out_channel)][int(row.in_channel)]))
	max_width_indices = get_max_edge_widths(widths)
	while len(names) < num_layers:
		edge_positions.append({'X':[],'Y':[],'Z':[]})
		colors.append([])
		widths.append([])
		weights.append([])
		names.append([])  
	return edge_positions, colors,widths,weights,names, max_width_indices


def get_edge_from_curvenumber(curvenum, edge_names, num_layers):
	edgenum = curvenum-(1+num_layers)
	curve=0
	for layer in range(len(edge_names)):
		for i in range(len(edge_names[layer])):
			if curve==edgenum:
				return layer, i, edge_names[layer][i]
			curve+=1
	return None,None,None


def edgename_2_edge_figures(edgename, image_name, kernels, activations, params):  #returns truth value of valid edge and kernel if valid
	#import pdb; pdb.set_trace()
	valid,from_layer,to_layer,from_within_id,to_within_id  = check_edge_validity(edgename,params)
	if valid:
		kernel=None
		in_map=None
		out_map=None
		if kernels is not None:
			kernel = kernels[to_layer][to_within_id][from_within_id]
			kernel = np.flip(kernel,0)
		if activations is not None and image_name is not None:
			if from_layer == 'img':
				#in_map = get_channelwise_image(image_name,from_within_id)
				in_map = activations['edges_in'][image_name][0][from_within_id]
			else:
				####!!!!!!!! This needs to be put through activation function (relu)
				#in_map = activations['nodes'][from_layer][list_of_input_images.index(image_name)][from_within_id]
				in_map = activations['edges_in'][image_name][from_layer+1][from_within_id]
			in_map = np.flip(in_map,0)
			out_map = activations['edges_out'][image_name][to_layer][to_within_id][from_within_id]
			out_map = np.flip(out_map,0)
		return kernel,in_map,out_map
		
	else:
		return None,None,None  



#RANK FUNCTIONS


#NETWORK GRAPH FUNCTIONS

def gen_networkgraph_traces(state,params):
	print('building graph from browser "state"')
	layer_colors = params['layer_colors']
	layer_nodes = params['layer_nodes']
	
	#add imgnodes
	colors = deepcopy(state['imgnode_colors'])
	if not str(state['node_select_history'][-1]).isnumeric():
		colors[state['imgnode_names'].index(state['node_select_history'][-1])] = 'rgba(0,0,0,1)'
	imgnode_trace=go.Scatter3d(x=state['imgnode_positions']['X'],
			   y=params['imgnode_positions']['Y'],
			   z=params['imgnode_positions']['Z'],
			   mode='markers',
			   name='image channels',
			   marker=dict(symbol='square',
							 size=8,
							 opacity=.99,
							 color=colors,
							 #colorscale='Viridis',
							 line=dict(color='rgb(50,50,50)', width=.5)
							 ),
			   text=params['imgnode_names'],
			   hoverinfo='text'
			   )

	imgnode_traces = [imgnode_trace]


	node_traces = []
	select_layer,select_position = None,None
	if str(state['node_select_history'][-1]).isnumeric():
		select_layer,select_position, select_layer_name = nodeid_2_perlayerid(state['node_select_history'][-1],params)
	for layer in range(len(layer_nodes)):
		#add nodes
		colors = deepcopy(state['node_colors'][layer])
		if layer == select_layer:
			colors[select_position] = 'rgba(0,0,0,1)'

		within_layer_ids = list(range(len(state['node_positions'][layer]['X'])))
		scores = state['node_weights'][layer]
		ids = layer_nodes[layer][1]
		#print(np.dstack((ids,within_layer_ids,scores)).shape)
		#print(np.dstack((ids,within_layer_ids,scores)))
		# hovertext = ['<b>%{id}</b>' +
		# 			'<br><i>layerwise ID</i>: %{within_layer_id}'+
		# 			'<br><i>Score</i>: %{score}<br>'
  		# 			 for id, within_layer_id, score in
		# 			 zip(ids, within_layer_ids, scores)]
		#print(hovertext) 
		node_trace=go.Scatter3d(x=state['node_positions'][layer]['X'],
				   y=state['node_positions'][layer]['Y'],
				   z=state['node_positions'][layer]['Z'],
				   mode='markers',
				   name=layer_nodes[layer][0],
				   marker=dict(symbol='circle',
								 size=6,
								 opacity=.99,
								 color=colors,
								 #colorscale='Viridis',
								 line=dict(color='rgb(50,50,50)', width=.5)
								 ),
				   text=ids,
				   #customdata = np.dstack((ids,within_layer_ids,scores)),
				   customdata = np.stack((ids,within_layer_ids,scores),axis=-1),
				   hovertemplate =	'<b>%{customdata[0]}</b>' +
		 					'<br><i>layerwise ID</i>: %{customdata[1]}'+
		 					'<br><i>Score</i>: %{customdata[2]:.3f}<br>'
				   #hoverinfo='text'
				   )

		node_traces.append(node_trace)


	edge_traces = []
	for layer in range(len(state['edge_positions'])):  
		legendgroup = layernum2name(layer ,title = 'edges')
		for edge_num in range(len(state['edge_positions'][layer]['X'])):  
		#add edges      
			color = deepcopy(state['edge_colors'][layer][edge_num])
			if state['edge_names'][layer][edge_num] == state['edge_select_history'][-1]:
				color = 'rgba(0,0,0,1)'
			showlegend = False
			if state['max_edge_width_indices'][layer] == edge_num:
				showlegend = True
			edge_trace=go.Scatter3d(x=state['edge_positions'][layer]['X'][edge_num],
									y=state['edge_positions'][layer]['Y'][edge_num],
									z=state['edge_positions'][layer]['Z'][edge_num],
									legendgroup=legendgroup,
									showlegend=showlegend,
									name=layer_nodes[layer][0],
									mode='lines',
									#line=dict(color=edge_colors_dict[layer], width=1.5),
									line=dict(color=color, width=state['edge_widths'][layer][edge_num]),
									text = state['edge_names'][layer][edge_num],
									hoverinfo='text'
									)
			edge_traces.append(edge_trace)


	combined_traces = imgnode_traces+node_traces+edge_traces
	return combined_traces



def load_cnn_gui_params(prepped_model_path,deepviz_neuron=None,deepviz_edge=False,show_ablations=False,show_act_map_means=False,show_image_manip = False,
						colorscale = 'RdBu',node_size=12,edge_size=1,max_node_inputs=20):

	full_prepped_model_path = os.path.abspath(prepped_model_path)
	update_sys_path(full_prepped_model_path)
	import prep_model_params_used as prep_model_params

	prepped_model_folder = prepped_model_path.split('/')[-1]

	params = {}
	params['prepped_model'] = prepped_model_folder
	params['prepped_model_path'] = full_prepped_model_path

	#modules
	params['show_ablations'] = show_ablations    #show network ablations modules
	params['show_act_map_means'] = show_act_map_means #show mean value under activation maps
	params['show_image_manip'] = show_image_manip


	#deepviz
	if deepviz_neuron is None:
		params['deepviz_neuron'] = prep_model_params.deepviz_neuron
	else:
		params['deepviz_neuron'] = deepviz_neuron
	params['deepviz_param'] = prep_model_params.deepviz_param
	params['deepviz_optim'] = prep_model_params.deepviz_optim
	params['deepviz_transforms'] = prep_model_params.deepviz_transforms
	params['deepviz_image_size'] = prep_model_params.deepviz_image_size
	params['deepviz_edge'] = deepviz_edge

	#backend
	params['device'] = prep_model_params.device
	params['input_image_directory'] = prep_model_params.input_img_path+'/'
	params['preprocess'] = prep_model_params.preprocess     #torchvision transfrom to pass input images through
	params['label_file_path'] = prep_model_params.label_file_path
	params['criterion'] = prep_model_params.criterion
	params['rank_img_path'] = prep_model_params.rank_img_path
	params['num_workers'] = prep_model_params.num_workers
	params['seed'] = prep_model_params.seed
	params['batch_size'] = prep_model_params.batch_size



	#aesthetic 
	params['colorscale'] = colorscale
	params['node_size'] = node_size
	params['edge_size'] = edge_size
	params['max_node_inputs'] = max_node_inputs 
	params['max_edge_weight'] = 1  #should get rid of this
	params['window_size'] = 'large' #this too

	params['layer_colors'] = ['rgba(31,119,180,', 
                          'rgba(255,127,14,',
                          'rgba(44,160,44,', 
                          'rgba(214,39,40,',
                          'rgba(39, 208, 214,', 
                          'rgba(242, 250, 17,',
                          'rgba(196, 94, 255,',
                          'rgba(193, 245, 5,',
                          'rgba(245, 85, 5,',
                          'rgba(5, 165, 245,',
                          'rgba(245, 5, 105,',
                          'rgba(218, 232, 23,',
                          'rgba(148, 23, 232,',
                          'rgba(23, 232, 166,']

	#misc graph data
	misc_data = pickle.load(open(full_prepped_model_path+'/misc_graph_data.pkl','rb'))
	params['layer_nodes'] = misc_data['layer_nodes']
	params['num_layers'] = misc_data['num_layers']
	params['num_nodes'] = misc_data['num_nodes']
	params['categories'] = misc_data['categories']
	params['num_img_chan'] = misc_data['num_img_chan']
	params['imgnode_positions'] = misc_data['imgnode_positions']
	params['imgnode_colors'] = misc_data['imgnode_colors']
	params['imgnode_names'] = misc_data['imgnode_names']
	params['ranks_data_path'] = full_prepped_model_path+'/ranks/'
	
	#input images
	params['input_image_directory'] = prep_model_params.input_img_path+'/'
	params['input_image_list'] = os.listdir(params['input_image_directory'])
	params['input_image_list'].sort()

	return params

def launch_cnn_gui(prepped_model,port=8050,params = None,deepviz_neuron=None,deepviz_edge=False,show_ablations=False,show_act_map_means=False,
					show_image_manip = False,colorscale = 'RdBu',node_size=12,edge_size=1,max_node_inputs=20,
					init_target_category = 'overall',init_rank_type = 'actxgrad',init_projection = 'MDS smooth',
					init_edge_threshold = [.7,1],init_node_threshold = [.4,1],download_images=True):
	'''
	This script generates a dash app for the viscnn exploratory GUI, to be launched 
	NOTE: Calling this function takes up a lot of memory, might want to clear space up before running it, (for example, with %reset in a jupyter notebook)


	args:
		prepped_model (required): Use the name of a folder inside the 'prepped_models' folder.
								  Prepped models available online that have not yet been downloaded
								  (alexnet, mnist, alexnet_sparse, vgg16) can also be put in this argument
								  and the appropriate files will be downloaded to the computer.
								  All this assumes the 'prepped_models' folder is in its 
								  original location (as downloaded from github), namely, 
								  in the root directory of the viscnn repo. If the 'prepped_models'
								  folder has been moved, specify the full path to your target folder
								  as this argument. 

		deepviz_neuron: if True use 'neuron' level visualizations, if False use 'channel' level
		deepviz_edge: if True, generate a feature vizualization for the output activations from edges. If False vizualize the node leading into and out of the edge instead
		device: use 'cuda:0', 'cuda:1' etc if you want to run the model on gpu, use 'cpu' otherwise
		params: Use 'None' to load params specified in 'prep_model_params_used.py' in the prepped_model folder. Otherwise set this to a custom parameter dictionary.

	'''

	#figure out where the prepped model is
	if '/' in prepped_model:
		prepped_model_path = os.path.abspath(prepped_model)
		prepped_model_folder = prepped_model.split('/')[-1]
	else:
		from viscnn import prepped_models_root_path
		prepped_model_path = prepped_models_root_path + '/' + prepped_model
		prepped_model_folder = 
		
	if not os.path.isdir(prepped_model_path):
		#try to download prepped_model from gdrive
		from subprocess import call
		if download_images:
			call('python download_from_gdrive.py %s'%prepped_model_folder,shell=True)
		else:
			call('python download_from_gdrive.py %s --dont-download-images'%prepped_model_folder,shell=True)




	#get prepped model and params
	if params is None:
		params = load_cnn_gui_params(prepped_model_path,deepviz_neuron=deepviz_neuron,deepviz_edge=deepviz_edge,show_ablations=show_ablations,show_act_map_means=show_act_map_means,
									 show_image_manip = show_image_manip,colorscale = colorscale,node_size=node_size,edge_size=edge_size,max_node_inputs=max_node_inputs)


	#load Model
	update_sys_path(prepped_model_path)
	import prep_model_params_used as prep_model_params
	model = prep_model_params.model
	_ = model.to(params['device']).eval()

	model_dis = dissect_model(deepcopy(prep_model_params.model),store_ranks=True,clear_ranks=True,device=params['device'])
	_ = model_dis.to(params['device']).eval()


	#GUI parameters initialization (these parameters can be set in the GUI, but what values should they be initialized to?)
	target_category = init_target_category     #category of images edges and nodes are weighted based on (which subgraph) 
	rank_type = init_rank_type      #weighting criterion (actxgrad, act, grad, or weight)
	projection = init_projection           #how nodes within a layer are projected into the 2d plane (MDS or Grid)
	edge_threshold = init_edge_threshold   #what range do edge ranks need to be in to be visualized
	node_threshold = init_node_threshold   #only relevant for hierarchical subgraph 

	#load nodes df
	print('loading nodes rank data')
	target_node = 'loss'
	categories_nodes_df = pd.read_csv('prepped_models/%s/ranks/categories_nodes_ranks.csv'%prepped_model_folder)
	target_nodes_df = categories_nodes_df.loc[categories_nodes_df['category']==target_category]
	target_nodes_df = minmax_normalize_ranks_df(target_nodes_df,params,weight=False)
	weight_nodes_df = pd.read_csv('prepped_models/%s/ranks/weight_nodes_ranks.csv'%prepped_model_folder)
	weight_nodes_df = minmax_normalize_ranks_df(weight_nodes_df,params,weight=True)
	node_colors,node_weights = gen_node_colors(target_nodes_df,rank_type,params) 

	#load node positions
	print('loading node position data')
	all_node_positions = pickle.load(open('./prepped_models/%s/node_positions.pkl'%prepped_model_folder,'rb'))
	if projection == 'Grid':
		node_positions = all_node_positions[projection]
	else:
		node_positions = all_node_positions[projection][rank_type]

	#Load Edge Kernels
	print('loading convolutional kernels')
	kernels = torch.load('prepped_models/%s/kernels.pt'%prepped_model_folder)['kernels']
	kernel_colors = torch.load('prepped_models/%s/kernels.pt'%prepped_model_folder)['kernel_colors']

	#load edges
	print('loading edge data')
	categories_edges_df = None
	if os.path.exists('prepped_models/%s/edge_ranks.csv'%prepped_model_folder):
		categories_edges_df = pd.read_csv('prepped_models/%s/ranks/categories_edges_ranks.csv'%prepped_model_folder)   #load edges
	if categories_edges_df is not None:
		#overall_edges_df = categories_edges_df.loc[categories_edges_df['category']=='overall']
		target_edges_df = categories_edges_df.loc[categories_edges_df['category']==target_category]
	else:
		#overall_edges_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_edges','overall_edges_rank.pt'))
		target_edges_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_edges','%s_edges_rank.pt'%target_category))
	target_edges_df = minmax_normalize_ranks_df(target_edges_df,params,weight=False)
	weight_edges_df = pd.read_csv('prepped_models/%s/ranks/weight_edges_ranks.csv'%prepped_model_folder)
	weight_edges_df = minmax_normalize_ranks_df(weight_edges_df,params,weight=True)    		
	edges_thresholded_df = get_thresholded_ranksdf(edge_threshold,rank_type,target_edges_df)	
	num_edges = len(target_edges_df)
	edges_df_columns = list(target_edges_df.columns)
	edge_positions, edge_colors, edge_widths, edge_weights, edge_names, max_edge_width_indices = gen_edge_graphdata(edges_thresholded_df, node_positions, rank_type, target_category,params,kernel_colors)

	#input image
	input_image_name = params['input_image_list'][0]
	input_image_size = params['deepviz_image_size']

	#receptive field
	receptive_fields = None
	if os.path.exists(params['prepped_model_path']+'/receptive_fields.pkl'):
		receptive_fields = pickle.load(open(params['prepped_model_path']+'/receptive_fields.pkl','rb'))
		
	#Format Node Feature Maps
	print('loading activation maps')

	all_activations = {'nodes':{},'edges_in':{},'edges_out':{}}
	if os.path.exists('prepped_models/%s/input_img_activations.pt'%prepped_model_folder):
		all_activations = torch.load('prepped_models/%s/input_img_activations.pt'%prepped_model_folder)

	#hidden state, stores python values within the html itself
	state = {'projection':projection,'rank_type':rank_type,'edge_positions':edge_positions,'edge_colors': edge_colors, 'edge_widths':edge_widths,'edge_names':edge_names,
			'edge_threshold':edge_threshold,'edge_weights':edge_weights,'max_edge_width_indices':max_edge_width_indices,
			'node_positions':node_positions,'node_colors':node_colors,'node_weights':node_weights,'node_threshold':node_threshold,'target_category':target_category,'target_node':'loss',
			'node_select_history':['0'],'edge_select_history':[edge_names[0][0]],'last_trigger':None,'input_image_name':input_image_name,
			'imgnode_positions':params['imgnode_positions'],'imgnode_colors':params['imgnode_colors'],'imgnode_names':params['imgnode_names']}


	#App Component Layouts
	display_dict = {True:'block',False:'none'}

	window_size_dict = {'large':{
                            'image':('14vw','14vw'),
                            'standard':('10vw','10vw'),
                            'standardwbar':('13vw','10vw'),
                            'kernel':('10vw','8vw'),
                            'node_inputs':(440,260)
                            },
                    'small':{
                            'image':('12vw','12vw'),
                            'standard':('8vw','8vw'),
                            'standardwbar':('6vw','4vw'),
                            'kernel':('8vw','6vw'),
                            'node_inputs':(320,200)
                            }
                   }

	figure_init = go.Figure()
	figure_init.add_trace(go.Scatter(
				x=[],
				y=[]))
	figure_init.update_layout(xaxis=dict(visible=False),
					yaxis=dict(visible=False),
					annotations = [dict(text="No Inputs",
										xref="paper",
										yref="paper",
										showarrow=False,
										font=dict(size=28))]
					)

	#Generate Network Graph
	combined_traces = gen_networkgraph_traces(state,params)
	network_graph_fig=go.Figure(data=combined_traces, layout=network_graph_layout)

	#external_stylesheets = ['https://codepen.io/amyoshino/pen/jzXypZ.css']
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

	app = dash.Dash(external_stylesheets = external_stylesheets)

	if not os.path.exists(params['prepped_model_path']+'/cache/'):
		os.mkdir(params['prepped_model_path']+'/cache/')
	CACHE_CONFIG = {
		# try 'filesystem' if you don't want to setup redis
		'CACHE_TYPE': 'filesystem',
		'CACHE_DIR': params['prepped_model_path'] +'/cache/'}
	cache = Cache()
	cache.init_app(app.server, config=CACHE_CONFIG)
		
	styles = {
		'pre': {
			'border': 'thin lightgrey solid',
			'overflowX': 'scroll'
		}
	}

	theme =  {
		'dark': True,
		'detail': '#007439',
		'primary': '#00EA64',
		'secondary': '#6E6E6E',
	}

	app.layout = html.Div([
			html.Div(
				children = [
					
				html.Div(
					#Left side control panel
					children = [
					html.Label('Subgraph Controls', style={'fontSize': 18,'font-weight':'bold'}),
					html.Br(),
					html.Label('Input'),
					#dcc.Dropdown(
					#  id='weight-category',
					#  options=[{'label': i, 'value': i} for i in params['categories']],
					#   value=target_category
					#   ),
					dcc.Input(id='input-category',value=state['target_category']),
					html.Br(),
					html.Br(),
					html.Label('Output'),
					#dcc.Dropdown(
					#  id='weight-category',
					#  options=[{'label': i, 'value': i} for i in params['categories']],
					#   value=target_category
					#   ),
					dcc.Dropdown(
						id='target-node',
						options=[
						{'label': i, 'value': i} for i in ['loss']+[str(node) for node in list(range(params['num_nodes']))]
						],
						value=state['target_node']),
					html.Br(),
					html.Label('Subgraph Criterion'),
					dcc.Dropdown(
						id='subgraph-criterion',
						options=[
							{'label': 'Activations*Grads', 'value': 'actxgrad'},
							{'label': 'Activations', 'value': 'act'},
							{'label': 'Gradients', 'value': 'grad'},
							{'label': 'Weights', 'value': 'weight'},
							{'label': 'Hierarchical', 'value': 'hierarchical'}
							
						],
						value='actxgrad'
						),
					html.Br(),   
					html.Label('Layer Projection'),
					dcc.Dropdown(
						id = 'layer-projection',
						options=[
							{'label': 'MDS', 'value': 'MDS'},
							{'label': 'MDS smooth', 'value': 'MDS smooth'},
							{'label': 'Grid', 'value': 'Grid'},
							{'label': 'Deepviz UMAP smooth', 'value': 'Deepviz UMAP smooth'}
							#{'label': 'SOM', 'value': 'SOM'}
						],
						value='MDS smooth'
						),

					html.Br(),
					html.Label('Edge Thresholds'),
						dcc.RangeSlider(
							id='edge-thresh-slider',
							min=0,
							max=np.ceil(params['max_edge_weight']*10)/10,
							step=0.001,
							marks={i/10: str(i/10) for i in range(0,int(np.ceil(params['max_edge_weight']*10))+1,int(round(np.ceil(params['max_edge_weight']*10)/5)))},
							value=edge_threshold,
						),
					html.Label('Node Thresholds'),
						dcc.RangeSlider(
							id='node-thresh-slider',
							min=0,
							max=1,
							step=0.001,
							marks={i/10: str(i/10.0) for i in range(0,11)},
							value=node_threshold,
						),
					html.Br(),
					dcc.Input(id="extract-name", placeholder="extract file name"),
					html.Button('Extract Subgraph', id='extract-button')
					], className="two columns",
					),
					
				html.Div([
					dcc.Graph(
						id='network-graph',
						figure=network_graph_fig
					)
					], className= 'ten columns'
					),
				], className="row"
			),


					
			html.Div([
				html.Div([
				html.Label('Input Image', style={'fontSize': 18,'font-weight':'bold'}),
				dcc.Dropdown(id="dynamic-input-image-dropdown",value=params['input_image_list'][0]),
				#dcc.Dropdown(
				#    id='input-image-dropdown',
				#    options=[{'label': i, 'value': i} for i in params['input_image_list']+os.listdir(params['prepped_model_path']+'/visualizations/images/')],
				#    value=input_image_name
				#),
				html.Br(),
				dcc.Graph(
					id='img-actmap-graph',
					style={
				'width': window_size_dict[params['window_size']]['standard'][0],
				'height':window_size_dict[params['window_size']]['standard'][1]
					},
					figure=image2heatmap(params['input_image_directory']+input_image_name,input_image_layout),
					config={
							'displayModeBar': False
							}
				)
				], className = "two columns"),

				html.Div([
				html.Label('Node', style={'fontSize': 18,'font-weight':'bold'}),
				dcc.Dropdown(
					id='node-actmap-dropdown',
					options=[{'label': str(j), 'value': str(j)} for j in params['imgnode_names']]+[{'label': str(i), 'value': str(i)} for i in range(params['num_nodes'])],
					value='0'
				),
				html.Br(),
					
				html.Div(className='colorbarplot-ratio-box',
				children=[
				html.Div(className='square-ratio-box-inside',
				children=[
						dcc.Graph(
						id='node-actmap-graph',
						style={
					#'width': window_size_dict[params['window_size']]['standardwbar'][0],
					#'height':window_size_dict[params['window_size']]['standardwbar'][0]
						'width':'100%',
						'height':'100%',   
						},
						figure=figure_init,
						config={'responsive': True,
								'displayModeBar': False
								}
				)])]),    
					

				dcc.Checklist(
					id = 'relu-checkbox',
					options = [{'label':'relu','value':'relu'}],
					value = []
					
				),
				html.Div(id='node-sum', style={'whiteSpace': 'pre-line','display': display_dict[params['show_act_map_means']]}),
				#html.Br(),
				#html.Br(),

				dcc.Graph(
					id='node-deepviz-image',
					style={
				'width': window_size_dict[params['window_size']]['standard'][0],
				'height':window_size_dict[params['window_size']]['standard'][1]
					},
					figure=figure_init,
					config={
							'displayModeBar': False
							}
				)
				], className = "three columns"),
				
				html.Div([
				html.Label('Node Inputs', style={'fontSize': 18,'font-weight':'bold'}),
				html.Br(),
				html.Div(dcc.Graph(
					id='node-inputs-graph',
					figure=figure_init,
					config={
							'displayModeBar': False
							}
				),style={'overflowY': 'scroll', 'height': 500})
				], className = "three columns"),

				html.Div([
				html.Label('Edge', style={'fontSize': 18,'font-weight':'bold'}),    
				dcc.Input(
					id='edge-actmaps-input',value=state['edge_names'][0][0], type='text'),
				#html.Button(id='edge-kernel-button',n_clicks=0, children='Submit'),
				#html.Br(),
				#html.Br(),
				html.Label('Kernel'),
				dcc.Graph(
					id='edge-kernel-graph',
					style={
				'width': window_size_dict[params['window_size']]['kernel'][0],
				'height':window_size_dict[params['window_size']]['kernel'][1]
					},
					figure=go.Figure(data=go.Heatmap(
													z = edgename_2_edge_figures(state['edge_names'][0][0], input_image_name, kernels, None,params)[0],
													colorscale=params['colorscale'],
													reversescale=True,
													zmid=0),
									layout=kernel_layout
									),
					config={
							'displayModeBar': False
							}
				),
				#html.Br(),
				html.Br(),
				dcc.Graph(
				id='edge-deepviz-image',
				style={
				'width': window_size_dict[params['window_size']]['standard'][0],
				'height':window_size_dict[params['window_size']]['standard'][1]
				},
				figure=figure_init,
				config={
						'displayModeBar': False
						}
				)
				], className = "two columns"),


				html.Div([
				html.Label('Edge Input'),
				#html.Br(),
				html.Div(className='colorbarplot-ratio-box',
				children=[
				html.Div(className='square-ratio-box-inside',
				children=[
						dcc.Graph(
						id='edge-inmap-graph',
						style={
					#'width': window_size_dict[params['window_size']]['standardwbar'][0],
					#'height':window_size_dict[params['window_size']]['standardwbar'][0]
						'width':'100%',
						'height':'100%',   
						},
						figure=figure_init,
						config={'responsive': True,
								'displayModeBar': False
								}
				)])]),
					
				html.Div(id='edgein-sum', style={'whiteSpace': 'pre-line','display': display_dict[params['show_act_map_means']]}),
				#html.Br(),
				html.Br(),
				#html.Br(),

				html.Label('Edge Output'),
				#html.Br(),
				html.Div(className='colorbarplot-ratio-box',
				children=[
				html.Div(className='square-ratio-box-inside',
				children=[
						dcc.Graph(
						id='edge-outmap-graph',
						style={
					#'width': window_size_dict[params['window_size']]['standardwbar'][0],
					#'height':window_size_dict[params['window_size']]['standardwbar'][0]
						'width':'100%',
						'height':'100%',   
						},
						figure=figure_init,
						config={'responsive': True,
								'displayModeBar': False
								}
				)])]),
					
				html.Div(id='edgeout-sum', style={'whiteSpace': 'pre-line','display': display_dict[params['show_act_map_means']]}),

				], className = "two columns")


			], className= 'row'
			),
		
		
		html.Div([
				html.Div([
				html.Label('Image Manipulations', style={'fontSize': 18,'font-weight':'bold'}),
				html.Br(),
				html.Label('rotation'),
				dcc.Slider(
					id='image-rotation-slider',
					min=0,
					max=350,
					step=10,
					marks={
							0:   '0°',
							20:  '20°',
							40:  '40°',
							60:  '60°',
							80:  '80°',
							100: '100°',
							120: '120°',
							140: '140°',
							160: '160°',
							180: '180°',
							200: '200°',
							220: '220°',
							240: '240°',
							260: '260°',
							280: '280°',
							300: '300°',
							320: '320°',
							340: '240°',
							},
					included=False,
					value=0,
				),
				html.Br(),
				html.Label('scaling'),
				dcc.Slider(
					id='image-scaling-slider',
					min=-10,
					max=10,
					step=1,
					marks={
							-8:  '.33',
							-6:  '.4',
							-4: '.5',
							-2: '.67',
							0: '1',
							2: '1.5',
							4: '2',
							6: '2.5',
							8: '3',
							},
					included=False,
					value=0,
				),            
				html.Br(),
				html.Label('colors'),

						html.Label('R',style={'fontSize': 10,'font-weight':'italic'}),
						dcc.Slider(
							id='image-r-slider',
							min=-1,
							max=1,
							step=.05,
							marks={
									-1:'-1',
									-.8:'-.8',
									-.6:'-.6',
									-.4:'-.4',
									-.2:'-.2',
									0:'0',
									.2:'.2',
									.4:'.4',
									.6:'.6',
									.8:'.8',
									1:'1',
									},
							included=False,
							value=0,
						),
	

						html.Label('G',style={'fontSize': 10,'font-weight':'italic'}),
						dcc.Slider(
							id='image-g-slider',
							min=-1,
							max=1,
							step=.05,
							marks={
									-1:'-1',
									-.8:'-.8',
									-.6:'-.6',
									-.4:'-.4',
									-.2:'-.2',
									0:'0',
									.2:'.2',
									.4:'.4',
									.6:'.6',
									.8:'.8',
									1:'1',
									},
							included=False,
							value=0,
						),
		

						html.Label('B',style={'fontSize': 10,'font-weight':'italic'}),
						dcc.Slider(
							id='image-b-slider',
							min=-1,
							max=1,
							step=.05,
							marks={
									-1:'-1',
									-.8:'-.8',
									-.6:'-.6',
									-.4:'-.4',
									-.2:'-.2',
									0:'0',
									.2:'.2',
									.4:'.4',
									.6:'.6',
									.8:'.8',
									1:'1',
									},
							included=False,
							value=0,
						)
			
				], className = "three columns", style= {'display': display_dict[params['show_image_manip']]} ),
					
					
				html.Div([
				html.Label('Feature Visualizations', style={'fontSize': 18,'font-weight':'bold'}),
				html.Br(),
				html.Div( style=dict(display='flex'),
					children = [     
						daq.ToggleSwitch(
							id='featviz-nodeedge-toggle',
							label=['node','edge    '],
							style={'float': 'right','margin': 'auto'}
							#labelPosition='bottom'
						), 
						html.Label(''),
						daq.ToggleSwitch(
							id='featviz-channelneuron-toggle',
							label=['channel','neuron    '],
							style={'float': 'right','margin': 'auto'}
							#labelPosition='bottom'
						),
						html.Label(''),
						daq.ToggleSwitch(
							id='featviz-positivenegative-toggle',
							label=['positive','negative    '],
							style={'float': 'right','margin': 'auto'}
							#labelPosition='bottom'
						)
					]),
				html.Br(),
				dcc.Graph(
				id='featviz-image',
				style={
				'width': window_size_dict[params['window_size']]['image'][0],
				'height':window_size_dict[params['window_size']]['image'][1]
				},
				figure=figure_init,
				config={
						'displayModeBar': False
						}
				),
				html.Button('Generate', id='featviz-button')
				#html.Button('Generate', id='gen-featviz-button')
				], className= "five columns", style= {'display': display_dict[params['show_ablations']]} ),
			
			
			
				html.Div([
				html.Label('Model Ablations', style={'fontSize': 18,'font-weight':'bold'}),
				dcc.Textarea(
					id='ablations-textarea',
					value='',
					style={'width': '70%', 'height': 300}),
				html.Button('Ablate', id='ablate-model-button')
				], className= "four columns"),
			
			], className="row", style= {'display': display_dict[params['show_ablations']]} 
			),

		#hidden divs for storing intermediate values     
		# The memory store reverts to the default on every page refresh
		dcc.Store(id='memory',data=state),
		# The local store will take the initial data
		# only the first time the page is loaded
		# and keep it until it is cleared.
		#dcc.Store(id='local', storage_type='local'),
		# Same as the local store but will lose the data
		# when the browser/tab closes.
		#dcc.Store(id='session', storage_type='session',data=state),
		

		# hidden signal value
		html.Div(id='input-image-signal',  style={'display': 'none'}),
		html.Div(id='target-signal', style={'display': 'none'},children = [state['target_category'],state['target_node']]),
		html.Div(id='ablations-signal',  style={'display': 'none'}, children = []),
		html.Div(id='extract-signal', style={'display': 'none'})
	])



	# perform expensive computations in this "global store"
	# these computations are cached in a globally available
	# 'cached' folder in the prepped_models/[model] folder
	@cache.memoize()
	def activations_store(image_name,ablation_list):

		print('Updating cached activations with {}'.format(image_name))
		activations = get_model_activations_from_image(get_image_path(image_name,params)[1], model_dis, params)
		
		return activations

	@app.callback(Output('input-image-signal', 'children'), 
				[Input('dynamic-input-image-dropdown', 'value'),
				Input('ablations-signal', 'children')])
	def update_activations_store(image_name,ablation_list):
		# compute value and send a signal when done
		activations_store(image_name,ablation_list)
		return image_name


	@cache.memoize()
	def ranksdf_store(target_category, target_node,ablation_list,model_dis=model_dis):
		print('Updating cached rank dfs with {}'.format(target_category))
		model_dis = clear_ranks_across_model(model_dis)
		target_type = image_category_or_contrast(target_category,params)
		target_category_nodes_df = None
		target_category_edges_df = None
		if target_type == 'category' and target_node == 'loss' and ablation_list == []:
			#edges
			if categories_edges_df is not None:
				if len(categories_edges_df.loc[categories_edges_df['category']==target_category]) > 0:
					target_category_edges_df = categories_edges_df.loc[categories_edges_df['category']==target_category]
			if target_category_edges_df is None:
				target_category_edges_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_edges','%s_edges_rank.pt'%target_category))   
			#node
			if categories_nodes_df is not None:
				if len(categories_nodes_df.loc[categories_nodes_df['category']==target_category]) > 0:
					target_category_nodes_df = categories_nodes_df.loc[categories_nodes_df['category']==target_category]
			if target_category_nodes_df is None:
				target_category_nodes_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_nodes','%s_nodes_rank.pt'%target_category))
		elif target_type == 'category':
			target_category_nodes_df,target_category_edges_df = rank_dict_2_df(get_model_ranks_for_category(target_category, target_node, model_dis,params))
		elif target_type == 'input_image':
			target_category_nodes_df,target_category_edges_df = rank_dict_2_df(get_model_ranks_from_image(get_image_path(target_category,params)[1],target_node, model_dis, params))

		else:  #contrast
			target_category_nodes_df,target_category_edges_df = contrast_str_2_dfs(target_category,target_node,model_dis,params,ablation_list)
		print('FROM RANKS DF STORE')
		print(target_category_edges_df)
		return target_category_nodes_df,target_category_edges_df

	@app.callback(Output('target-signal', 'children'), 
				[Input('input-category', 'value'),
				Input('target-node','value'),
				Input('ablations-signal', 'children')])
	def update_ranksdf_store(target_category,target_node,ablation_list):
		# compute value and send a signal when done
		print('update ranksdf_store triggered')
		ranksdf_store(target_category,target_node,ablation_list)
		return [target_category,target_node]



	####Call Back Functions

	#Ablations
	@app.callback(Output('ablations-signal', 'children'), 
				[Input('ablate-model-button', 'n_clicks')],
				[State('ablations-textarea','value')])
	def update_ablations(n_clicks,text,model_dis=model_dis):
		# compute value and send a signal when done
		ablation_list = ablation_text_2_list(text, params)
		ablate_model_with_list(ablation_list,model_dis,params)
		return ablation_list


	#Hidden State
	@app.callback(Output('memory', 'data'),
				[Input('target-signal', 'children'),
				Input('node-actmap-dropdown', 'value'),
				Input('edge-actmaps-input', 'value'),
				Input('edge-thresh-slider','value'),
				Input('node-thresh-slider','value'),
				Input('layer-projection','value'),
				Input('subgraph-criterion','value')],
				[State('memory', 'data'),
				State('ablations-signal', 'children')])
	def update_store(target,node_value,edge_value,edge_threshold,node_threshold,projection,rank_type,state,ablation_list):
		print('CALLED: update_store\n')
		ctx = dash.callback_context
		if not ctx.triggered:
			raise Exception('no figure updates yet')
		else:
			trigger = ctx.triggered[0]['prop_id']
		state['last_trigger'] = trigger  #store the last trigger of state change in state
		print('TRIGGER %s'%trigger)
		
		hierarchical = False
		if rank_type == 'hierarchical':
			hierarchical = True
			rank_type = 'actxgrad'
			
		target_category,target_node = target[0],target[1]
		#fetch select edges DF
		if trigger in ['target-signal.children','edge-thresh-slider.value','node-thresh-slider.value','layer-projection.value','subgraph-criterion.value']:
			if rank_type == 'weight':
				target_edges_df = weight_edges_df
				target_nodes_df = weight_nodes_df
				weight=True
			else:   
				target_nodes_df,target_edges_df = ranksdf_store(target_category,target_node,ablation_list)
				weight=False   
			target_edges_df = minmax_normalize_ranks_df(target_edges_df,params,weight=weight)
			target_nodes_df = minmax_normalize_ranks_df(target_nodes_df,params,weight=weight)

			if hierarchical:
				#nodes_thresholded_df = get_thresholded_ranksdf(node_threshold,rank_type, target_nodes_df)
				#filter_edges_df = filter_edges_by_nodes(target_edges_df,nodes_thresholded_df)
				#edges_thresholded_df = get_thresholded_ranksdf(edge_threshold,rank_type,filter_edges_df)
				#edges_thresholded_df = hierarchically_threshold_edges(edge_threshold,rank_type,target_edges_df,nodes_thresholded_df)
				print('finding hierarchical subgraph')
				start = time.time()
				nodes_thresholded_df,edges_thresholded_df = hierarchical_accum_threshold(node_threshold[0],edge_threshold[0],rank_type,target_edges_df,target_nodes_df,ascending=False)
				print('time: %s'%str(time.time() - start))
				print('found %s nodes and %s edges'%(str(len(nodes_thresholded_df)),str(len(edges_thresholded_df))))
				#node_minmax = node_threshold
				node_min = {}
				for layer in target_nodes_df['layer'].unique():
					if len(nodes_thresholded_df.loc[nodes_thresholded_df['layer']==layer]) > 1:
						node_min[layer] = nodes_thresholded_df.loc[nodes_thresholded_df['layer']==layer][rank_type+'_rank'].min()
					else:
						node_min[layer] = None

			else: 
				nodes_thresholded_df = None
				edges_thresholded_df = get_thresholded_ranksdf(edge_threshold,rank_type,target_edges_df)
				node_min = None

		if trigger == 'target-signal.children':
			print('changing target category to %s'%target_category)
			#print(target_nodes_df)
			state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params,node_min=node_min)
			#state['max_edge_weight'] = get_max_edge_weight(target_category)
			state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params,kernel_colors)

		elif trigger == 'node-actmap-dropdown.value' or trigger == 'edge-actmaps-input.value':
			state['last_trigger'] = 'selection_change'
			print(edge_value)
			#update node if button value different than store value
			if state['node_select_history'][-1] != node_value:
				print('changing selected node to %s'%node_value)
				state['node_select_history'].append(node_value)
				if len(state['node_select_history']) > 10:
					del state['node_select_history'][0] 
			#update edge if button value different than store value
			if state['edge_select_history'][-1] != edge_value and check_edge_validity(edge_value.strip(),params)[0]:
				print('changing selected edge to %s'%edge_value)
				state['edge_select_history'].append(edge_value)
				print(state['edge_select_history'])
				if len(state['edge_select_history']) > 10:
					del state['edge_select_history'][0]              

		elif trigger == 'edge-thresh-slider.value':
			print('changing edge thresholds to %s - %s'%(edge_threshold[0],edge_threshold[1]))
			state['edge_threshold'] == edge_threshold
			print('found %s edges'%len(edges_thresholded_df))
			state['edge_positions'], state['edge_colors'], state['edge_widths'], state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params,kernel_colors)
		
		elif trigger == 'node-thresh-slider.value':
			print('changing node thresholds to %s - %s'%(node_threshold[0],node_threshold[1]))
			state['node_threshold'] == node_threshold
			print('found %s nodes'%len(nodes_thresholded_df))
			state['edge_positions'], state['edge_colors'], state['edge_widths'], state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params,kernel_colors)
			state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params,node_min=node_min)
			
			
		elif trigger == 'layer-projection.value':
			print('changing layer projection to %s\n'%projection)
			state['projection']=projection
			if projection == 'Grid' or projection == 'Deepviz UMAP smooth':
				node_positions = all_node_positions[projection]
			else:
				node_positions = all_node_positions[projection][rank_type]
			state['node_positions'] = node_positions
			state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params,kernel_colors)

		elif trigger == 'subgraph-criterion.value':
			print('changing weighting criterion to %s\n'%rank_type)
			state['rank_type']=rank_type
			state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params,node_min=node_min)
			#state['node_positions']=format_node_positions(projection=projection,rank_type=rank_type)
			state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params,kernel_colors)

		else:
			raise Exception('unknown trigger: %s'%trigger)    
		return state


	#Network Graph Figure
	@app.callback(
		Output('network-graph', 'figure'),
		[Input('memory', 'data')],
		[State('network-graph','figure')])
	def update_figure(state, fig):
		#network_graph_layout['uirevision'] = True
		print('CALLED: update_figure\n')
		print(state['edge_threshold'])
		print(state['edge_select_history'])
		print(state['node_select_history'])
		if state['last_trigger'] == 'selection_change':   #minimal updates
			#hightlight edge
			print('updating edge highlight to %s'%state['edge_select_history'][-1])
			#if len(state['edge_select_history']) >1:
			#if state['edge_select_history'][-1] != state['edge_select_history'][-2]:  #didnt click same point
			flat_edge_names = [item for sublist in state['edge_names'] for item in sublist]
			flat_edge_colors = [item for sublist in state['edge_colors'] for item in sublist]
			try:  #update current edge if it exists to black
				#print(flat_edge_names)
				fig['data'][flat_edge_names.index(state['edge_select_history'][-1])+params['num_layers']+1]['line']['color'] = 'rgba(0,0,0,1)'
			except:
				print('select edge, %s,  not recolored as no longer shown'%state['edge_select_history'][-1])
			if len(state['edge_select_history']) > 1: #there is a previous edge to unselect
				try: #recolor previous edge if it exists from black
					fig['data'][flat_edge_names.index(state['edge_select_history'][-2])+params['num_layers']+1]['line']['color'] = flat_edge_colors[flat_edge_names.index(state['edge_select_history'][-2])]
				except:
					print('previous edge, %s,  not recolored as no longer shown'%state['edge_select_history'][-2])
			#highlight node
			print('updating node highlight to %s'%state['node_select_history'][-1])
			#if len(state['node_select_history']) >1:
			#    if state['node_select_history'][-1] != state['node_select_history'][-2]: 
					#update current node color to black
			if str(state['node_select_history'][-1]).isnumeric():  #if normal node
				select_layer,select_position,select_layer_name = nodeid_2_perlayerid(state['node_select_history'][-1],params)
				fig['data'][select_layer+1]['marker']['color'][select_position] = 'rgba(0,0,0,1)'
			else:   #imgnode
				fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-1])] = 'rgba(0,0,0,1)'
			#update previous node color to its usual color
			if len(state['node_select_history']) > 1: #there is a previous node to unselect
				if str(state['node_select_history'][-2]).isnumeric():  #if normal node
					prev_select_layer,prev_select_position,prev_select_layer_name = nodeid_2_perlayerid(state['node_select_history'][-2],params)
					print(prev_select_layer,prev_select_position,prev_select_layer_name)
					fig['data'][prev_select_layer+1]['marker']['color'][prev_select_position] = state['node_colors'][prev_select_layer][prev_select_position]
				else:   #imgnode
					fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-2])] = state['imgnode_colors'][fig['data'][0]['text'].index(state['node_select_history'][-2])]
			#fig['layout']['uirevision']=True   
			return fig    
		else:   #regenerate full traces
			combined_traces = gen_networkgraph_traces(state,params)
			fig['data'] = combined_traces
			#layout = network_graph_layout
			#layout['uirevision'] = True
			return fig

	#Node Actmap Dropdown
	@app.callback(
		Output('node-actmap-dropdown', 'value'),
		[Input('network-graph', 'clickData')],
		[State('node-actmap-dropdown', 'value')])
	def switch_node_actmap_click(clickData,current_value):
		print('CALLED: switch_node_actmap_click')
		if clickData is None:
			return current_value 
			#raise Exception('no click data')
		if int(clickData['points'][0]['curveNumber']) > params['num_layers']:
			return current_value
			#raise Exception('edge was clicked')
		return clickData['points'][0]['text']

	#Edge Actmaps Input
	@app.callback(
		Output('edge-actmaps-input', 'value'),
		[Input('network-graph', 'clickData')],
		[State('edge-actmaps-input', 'value'),
		State('memory', 'data')])
	def switch_edge_actmaps_click(clickData,current_value,state):
		print('CALLED: switch_edge_actmaps_click')
		if clickData is None:
			return current_value
			#raise Exception('no click data')
		if int(clickData['points'][0]['curveNumber']) <= params['num_layers']:
			return current_value
			#raise Exception('node was clicked')
		return get_nth_element_from_nested_list(state['edge_names'],int(clickData['points'][0]['curveNumber'])-(params['num_layers']+1))


	#Node actmap graph
	@app.callback(
		Output('node-actmap-graph', 'figure'),
		[Input('node-actmap-dropdown', 'value'),
		Input('relu-checkbox','value'),
		Input('input-image-signal', 'children')],
		[State('ablations-signal', 'children')])
	def update_node_actmap(nodeid,relu_checked,image_name,ablation_list):       #EDIT: needs support for black and white images
		print('CALLED: update_node_actmap')
		#import pdb; pdb.set_trace()
		layer, within_id,layer_name = nodeid_2_perlayerid(nodeid,params)
		#fetch activations
		
		if image_name in all_activations['nodes'] and ablation_list == []:
			activations = all_activations
		else:
			activations  = activations_store(image_name,ablation_list)
			
		if layer == 'img': #code for returning color channel as activation map
			#np_chan_im = get_channelwise_image(image_name,state['imgnode_names'].index(nodeid),params['input_image_directory']=params['input_image_directory'])
			np_chan_im = activations['edges_in'][image_name][0][within_id]
			return go.Figure(data=go.Heatmap( z = np.flip(np_chan_im,0), name = nodeid,
											colorscale=params['colorscale'],
											reversescale=True,
											zmid=0),
							layout=node_actmap_layout) 
		act_map = activations['nodes'][image_name][layer][within_id]
		
		if relu_checked != []:
			act_map = relu(act_map)
			
		return go.Figure(data=go.Heatmap( z = np.flip(act_map,0),
										colorscale=params['colorscale'],
										reversescale=True,
										zmid=0,
										#zmin=-11,
										#zmax=14,
										colorbar = dict(thicknessmode = "fraction",thickness=.1)
										),
						layout=node_actmap_layout) 

	@app.callback(
		Output('node-sum', 'children'),
		[Input('node-actmap-graph', 'figure')])
	def update_node_sum(fig):
		mean = np.mean(fig['data'][0]['z'])
		return 'mean: %s'%str(mean)


	#Node deepviz graph
	@app.callback(
		Output('node-deepviz-image', 'figure'),
		[Input('node-actmap-dropdown', 'value')])
	def update_node_deepviz(nodeid):       #EDIT: needs support for black and white images
		print('CALLED: update_node_deepviz')
		layer,within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)    
		if layer == 'img': 
			return figure_init
		image_name = fetch_deepviz_img(model,str(nodeid),params)
		image_path = params['prepped_model_path']+'/visualizations/images/'+image_name
		return image2plot(image_path,input_image_layout)
		

	#Edge deepviz graph
	@app.callback(
		Output('edge-deepviz-image', 'figure'),
		[Input('edge-actmaps-input', 'value')])
	def update_edge_deepviz(edgename):       #EDIT: needs support for black and white images
		print('CALLED: update_edge_deepviz')
		#layer,within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)    
		#if layer == 'img': 
		#    return figure_init
		if params['deepviz_edge']:
			image_name = fetch_deepviz_img(model_dis,edgename,params)
			image_path = params['prepped_model_path']+'/visualizations/images/'+image_name
			return image2plot(image_path,input_image_layout)
		else:
			image_in = fetch_deepviz_img(model,edgename.split('-')[0],params)
			image_out = fetch_deepviz_img(model,edgename.split('-')[1],params)
			image_in_path = params['prepped_model_path']+'/visualizations/images/'+image_in
			image_out_path = params['prepped_model_path']+'/visualizations/images/'+image_out
			combined = combine_images([image_in_path,image_out_path])
			return image2plot(combined,double_image_layout)
		

	#Node inputs actmap graph
	@app.callback(
		Output('node-inputs-graph', 'figure'),
		[Input('node-actmap-dropdown', 'value'),
		Input('input-image-signal', 'children'),
		Input('target-signal', 'children'),
		Input('subgraph-criterion','value')],
		[State('ablations-signal', 'children')])
	def update_node_inputs(nodeid,image_name,target,rank_type,ablation_list,model=model_dis,max_num = params['max_node_inputs']):       
		print('CALLED: update_node_inputs')
		
		hierarchical = False
		if rank_type == 'hierarchical':
			hierarchical = True
			rank_type = 'actxgrad'
		
		
		target_category,target_node = target[0],target[1]
		node_layer,node_within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)
		#fetch activations
		if image_name in all_activations['nodes'] and ablation_list == []:
			activations = all_activations
		else:
			activations = activations_store(image_name,ablation_list)
		#fetch edges df
		if rank_type == 'weight':
			target_edges_df = weight_edges_df
		else:
			target_edges_df = ranksdf_store(target_category,target_node,ablation_list)[1]
		#return no input if on input image node 
		if node_layer == 'img':
			fig = go.Figure()

			fig.add_trace(go.Scatter(
				x=[],
				y=[]))
			fig.update_layout(xaxis=dict(visible=False),
							yaxis=dict(visible=False),
							annotations = [dict(text="No Inputs",
												xref="paper",
												yref="paper",
												showarrow=False,
												font=dict(size=28))]
							)
			return fig

		all_node_edges_df = target_edges_df.loc[(target_edges_df['layer']==node_layer) & (target_edges_df['out_channel'] == node_within_layer_id)]
		#if sort_images:                      
		all_node_edges_df = all_node_edges_df.sort_values(by=[rank_type+'_rank'],ascending=False)
		top_node_edges_df = all_node_edges_df.head(max_num)
		fig = make_subplots(rows=len(top_node_edges_df)+1, cols=3)
		#print(top_node_edges_df)
		i=1
		for row in top_node_edges_df.itertuples():
			if node_layer == 0:
				edge_name = str(params['imgnode_names'][row.in_channel])+'-'+str(nodeid)
			else:
				edge_name = str(params['layer_nodes'][node_layer-1][1][row.in_channel])+'-'+str(nodeid)
			#add activation map
			fig.add_trace(
				go.Heatmap(z = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)[2],
							#zmin = -1,
							#zmax = 1,
							zmid=0,
							colorscale=params['colorscale'],
							reversescale=True,
							name = edge_name,
							coloraxis="coloraxis"
							#showscale = False,
							#colorbar = dict(lenmode='fraction',len=1/len(top_node_edges_df), 
							#                y=(i)/len(top_node_edges_df)-.01,
							#                thicknessmode = "fraction",thickness=.1,
							#                ypad=1
							#               )
							),
				row=i, col=2),
			#add kernel
			fig.add_trace(
				go.Heatmap(z = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)[0],
							#zmin = -1,
							#zmax = 1,
							zmid=0,
							colorscale=params['colorscale'],
							reversescale=True,
							name = edge_name+'_kernel',
							coloraxis="coloraxis2"
							#showscale = False,
							#colorbar = dict(lenmode='fraction',len=1/len(top_node_edges_df), 
							#                y=(i)/len(top_node_edges_df)-.01,
							#                thicknessmode = "fraction",thickness=.1,
							#                ypad=1
							#               )
							),
				row=i, col=3),

			#add visualization
			viz_img_name = fetch_deepviz_img_for_node_inputs(model,edge_name,params)
			viz_img_path = params['prepped_model_path']+'/visualizations/images/'+viz_img_name
			viz_img = Image.open(viz_img_path)
			#fig.add_trace(go.Image(z=viz_img,name=viz_img_name), row=i, col=1)
			fig.add_trace(go.Scatter(x=[],y=[]),row=i,col=1)
			fig.add_layout_image(
								source=viz_img,
								xref="x",
								yref="y",
								x=0,
								y=10,
								sizex=10,
								sizey=10,
								sizing="stretch",
								opacity=1,
								layer="below",
								row=i, col=1
								)
			fig.update_xaxes(visible=False,range=(0,10),showline=False,showgrid=False,showticklabels=False,row=i,col=1)
			fig.update_yaxes(visible=False,range=(0,10),showline=False,showgrid=False,showticklabels=False,row=i,col=1)
	


			i+=1
		
		fig.update_layout(height=window_size_dict[params['window_size']]['node_inputs'][1]*len(top_node_edges_df), 
						width=window_size_dict[params['window_size']]['node_inputs'][0],
						#yaxis=dict(scaleanchor="x", scaleratio=1/len(top_node_edges_df)),
						#title_text="Inputs to Node",
						#xaxis=dict(visible=False),
						#yaxis=dict(visible=False),
						coloraxis_showscale=False,
						coloraxis2 = dict(showscale=False,
											colorscale=params['colorscale'],
											reversescale=True,
											cmid=0,
											colorbar = dict(
															thicknessmode = "fraction",thickness=.05, 
															lenmode='fraction',len=.7)),
						margin=dict(
										l=0,
										r=0,
										b=0,
										t=0,
										pad=0)
						)
		fig.update_coloraxes(colorscale=params['colorscale'],reversescale=True,cmid=0,colorbar = dict(
																thicknessmode = "fraction",thickness=.05, 
																lenmode='fraction',len=.7)
							)
	#     fig.update_coloraxes2(colorscale='inferno',colorbar = dict(
	#                                                               thicknessmode = "fraction",thickness=.05, 
	#                                                               lenmode='fraction',len=.7)
	#                        )
		return fig







	#image graph
	@app.callback(
		Output('img-actmap-graph', 'figure'),
		[Input('dynamic-input-image-dropdown', 'value'),
		Input('node-actmap-graph','clickData'),
		Input('node-actmap-graph','figure')],
		[State('img-actmap-graph', 'figure'),
		State('node-actmap-dropdown', 'value')])
	def update_inputimg_actmap(image_name,click_data,node_actmap_fig,image_fig,nodeid): 
		print('CALLED: update_inputimg_actmap')
		#if os.path.exists(params['input_image_directory']+image_name):
		ctx = dash.callback_context
		if not ctx.triggered:
			raise Exception('no figure updates yet')
		else:
			trigger = ctx.triggered[0]['prop_id']
		if trigger == 'dynamic-input-image-dropdown.value':
			return image2plot(get_image_path(image_name,params)[1],input_image_layout)
		elif receptive_fields is None:
			return image2plot(get_image_path(image_name,params)[1],input_image_layout)
		elif click_data is None:
			return image2plot(get_image_path(image_name,params)[1],input_image_layout)
		else:
			#nodeid = node_actmap_fig['data'][0]['name']
			layer_name = nodeid_2_perlayerid(nodeid,params)[2]
			if layer_name == 'img':
				raise Exception('no receptive fields for input image actmap')
			heatmap_dim_y = len(node_actmap_fig['data'][0]['z'])
			heatmap_dim_x = len(node_actmap_fig['data'][0]['z'][0]) 
			x_click = click_data['points'][0]['x']
			y_click = heatmap_dim_y - click_data['points'][0]['y']-1
			print('x_click')
			print(x_click)
			print('y_click')
			print(y_click)
			recep_field = receptive_field_for_unit(receptive_fields, layer_name, (x_click,y_click))
			recep_field_normed = [[recep_field[0][0]*10/input_image_size,recep_field[0][1]*10/input_image_size],
								[recep_field[1][0]*10/input_image_size,recep_field[1][1]*10/input_image_size]]
			print('normalized')
			print(recep_field_normed)
			x_points = [recep_field_normed[0][0],recep_field_normed[0][0],recep_field_normed[0][1],recep_field_normed[0][1],recep_field_normed[0][0]]
			y_points = [10 - recep_field_normed[1][0],10 - recep_field_normed[1][1],10 - recep_field_normed[1][1],10 - recep_field_normed[1][0],10 - recep_field_normed[1][0]]
			print('x points')
			print(x_points)
			print('y points')
			print(y_points)
			image_fig['data'] = [{'mode': 'lines', 'x': x_points, 'y': y_points, 'type': 'scatter','line':{'color':'red'}}]
			return image_fig
			



		#else:
			#return image2plot(params['prepped_model_path']+'/visualizations/'+image_name,input_image_layout)
		
	# #image dropdown
	# @app.callback(
	#     Output('input-image-dropdown', 'options'),
	#     [Input('node-deepviz-image', 'figure'),
	#      Input('edge-deepviz-image', 'figure')])
	# def update_inputimg_dropdown(node_fig,edge_fig): 
	#     print('CALLED: update_inputimg_dropdown options')
	#     return [{'label': i, 'value': i} for i in params['input_image_list']+os.listdir(params['prepped_model_path']+'/visualizations/images/')]

	#dynamic dropdown
	@app.callback(
		dash.dependencies.Output("dynamic-input-image-dropdown", "options"),
		[dash.dependencies.Input("dynamic-input-image-dropdown", "search_value")],
	)
	def update_options(search_value):
		if not search_value:
			raise PreventUpdate
		return [{'label': i, 'value': i} for i in params['input_image_list']+os.listdir(params['prepped_model_path']+'/visualizations/images/') if search_value in i]





	#kernel
	@app.callback(
		Output('edge-kernel-graph', 'figure'),
		[Input('edge-actmaps-input','value')],
		[State('edge-kernel-graph','figure')])
	def update_edge_kernelmap(edge_name,figure):
		print('CALLED: update_edge_kernelmap')
		kernel,inmap,outmap = edgename_2_edge_figures(edge_name, None, kernels, None,params)
		if kernel is not None:
			return go.Figure(data=go.Heatmap(z = kernel,
											colorscale=params['colorscale'],
											reversescale=True,
											zmid=0,
											#zmin=-.5,
											#zmax=.5,
											colorbar = dict(thicknessmode = "fraction",thickness=.1)),
							layout=kernel_layout)
		else:
			return figure
					

	#edge in        
	@app.callback(
		Output('edge-inmap-graph', 'figure'),
		[Input('edge-actmaps-input','value'),
		Input('input-image-signal', 'children')],
		[State('edge-inmap-graph','figure'),
		State('ablations-signal', 'children')])
	def update_edge_inmap(edge_name,image_name,figure,ablation_list):
		print('CALLED: update_edge_inmap')
		#fetch activations
		if image_name in all_activations['nodes'] and ablation_list == []:
			activations = all_activations
		else:
			activations = activations_store(image_name, ablation_list)
			
		kernel,inmap,outmap = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)
		if inmap is not None:
			return go.Figure(data=go.Heatmap(z = inmap,
											zmid=0,
											colorscale=params['colorscale'],
											reversescale=True,
											#zmin=-2,zmax=2,
											colorbar = dict(thicknessmode = "fraction",thickness=.1)
											),
							layout=edge_inmap_layout)
		else:
			print('edge inmap error')
			return figure

	@app.callback(
		Output('edgein-sum', 'children'),
		[Input('edge-inmap-graph', 'figure')])
	def update_node_sum(fig):
		mean = np.mean(fig['data'][0]['z'])
		return 'mean: %s'%str(mean)    

	#edge out
	@app.callback(
		Output('edge-outmap-graph', 'figure'),
		[Input('edge-actmaps-input','value'),
		Input('input-image-signal', 'children')],
		[State('edge-outmap-graph','figure'),
		State('ablations-signal', 'children')])
	def update_edge_outmap(edge_name,image_name,figure, ablation_list):
		print('CALLED: update_edge_outmap')
		#fetch activations
		if image_name in all_activations['nodes'] and ablation_list == []:
			activations = all_activations
		else:
			activations = activations_store(image_name,ablation_list)
			
		kernel,inmap,outmap = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)
		if outmap is not None:
			return go.Figure(data=go.Heatmap(z = outmap,
											zmid=0,
											colorscale=params['colorscale'],
											reversescale=True,
											#zmin=-11,
											#zmax=14,
											colorbar = dict(thicknessmode = "fraction",thickness=.1)
											),
							layout=edge_outmap_layout)
		else:
			print('edge outmap error')
			return figure
			
	@app.callback(
		Output('edgeout-sum', 'children'),
		[Input('edge-outmap-graph', 'figure')])
	def update_node_sum(fig):
		mean = np.mean(fig['data'][0]['z'])
		return 'mean: %s'%str(mean)


	
	#feature viz graph
	@app.callback(
		Output('featviz-image', 'figure'),
		[Input('featviz-button','n_clicks')],
		[State('featviz-nodeedge-toggle', 'value'),
		State('featviz-channelneuron-toggle', 'value'),
		State('node-actmap-dropdown','value'),
		State('edge-actmaps-input','value')],
		prevent_initial_call=True)
	def update_featviz_image(n_clicks,edge,neuron,nodeid,edgeid):       #EDIT: needs support for black and white images
		print('CALLED: update_featviz')
		if edge:
			image_name = regen_visualization(model_dis,edgeid,neuron,params)
		else:
			layer,within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)    
			if layer == 'img': 
				return figure_init
			image_name = regen_visualization(model_dis,nodeid,neuron,params)
			
		image_path = params['prepped_model_path']+'/visualizations/images/'+image_name
		return image2plot(image_path,input_image_layout)




	#Extraction
	@app.callback(Output('extract-signal', 'children'), 
				[Input('extract-button', 'n_clicks')],
				[State('extract-name','value'),
				State('target-signal', 'children'),
				State('edge-thresh-slider','value'),
				State('node-thresh-slider','value'),
				State('subgraph-criterion','value'),
				State('memory', 'data'),
				State('ablations-signal', 'children')
				])
	def extract_subgraph_call(n_clicks,file_name,target,edge_threshold,node_threshold,rank_type,state,ablation_list,model_dis=model_dis):
		print('CALLED: extract_subgraph_call\n')
		if rank_type != 'hierarchical':
			raise Exception('subgraph criterion must be hierarchical to extract graph') 
		
		rank_type = 'actxgrad'
		target_category,target_node = target[0],target[1]
		#fetch select edges DF
	
		target_nodes_df,target_edges_df = ranksdf_store(target_category,target_node,ablation_list)

		target_edges_df = minmax_normalize_ranks_df(target_edges_df,params)
		target_nodes_df = minmax_normalize_ranks_df(target_nodes_df,params)

		print('finding hierarchical subgraph')
		start = time.time()
		nodes_thresholded_df,edges_thresholded_df = hierarchical_accum_threshold(node_threshold[0],edge_threshold[0],rank_type,target_edges_df,target_nodes_df,ascending=False)
		print('time: %s'%str(time.time() - start))
		print('found %s nodes and %s edges'%(str(len(nodes_thresholded_df)),str(len(edges_thresholded_df))))
		
		#make subgraph model
		sub_model = extract_subgraph(model,nodes_thresholded_df,edges_thresholded_df,params)
		save_object = {'model':sub_model,
					'node_df':nodes_thresholded_df,
					'edge_df':edges_thresholded_df,
					'gen_params':{'node_thresh':node_threshold[0],
									'edge_thresh':edge_threshold[0],
									'input':target_category,
									'output':str(target_node)}}
		if file_name[-3:] != '.pt':
			file_name_l = file_name.split('.')
			if len(file_name_l) == 1:
				file_name+='.pt'
			else:
				file_name = '.'.join(file_name_l[:-1])+'.pt'
		torch.save(save_object,'prepped_models/%s/subgraphs/models/%s'%(prepped_model_folder,file_name))
		print('SHOULD HAVE SAVED')

	app.run_server(port=port)

		
