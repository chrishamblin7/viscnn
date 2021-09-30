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
from prep_model_scripts.dissected_Conv2d import *
from prep_model_scripts.data_loading_functions import *


  


#INPUT IMAGE FUNCTIONS

def get_image_path(image_name,params):
	found = False
	path = None
	if image_name in params['input_image_list']:
		found = True
		path = params['input_image_directory']+'/'+image_name
	elif image_name in os.listdir(params['prepped_model_path']+'/visualizations/images/'):
		found = True
		path = params['prepped_model_path']+'/visualizations/images/'+image_name
	return found, path



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
