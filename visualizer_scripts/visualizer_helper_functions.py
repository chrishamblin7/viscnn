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
sys.path.insert(0, os.path.abspath('../prep_model_scripts/'))
from dissected_Conv2d import *
from data_loading_functions import *

#DATAFRAME FUNCTIONS

def rank_file_2_df(file_path):      #takes a node or edge 'rank.pt' file and turns it into a pandas dataframe, or takes the dict itself not file path
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

def rank_dict_2_df(ranks):      #takes a node or edge 'rank.pt' file and turns it into a pandas dataframe, or takes the dict itself not file path
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


###normalizing dataframe ranks

def normalize_ranks_df(df,norm,params,weight=False):    #this isnt used, might just be best to just min max normalize

	if norm == None:
		return df

	norm_funcs = {
				'std':lambda x: np.std(x),
				'mean':lambda x: np.mean(x),
				'max':lambda x: np.max(x),
				'l1':lambda x: np.sum(x),
				'l2':lambda x: np.sqrt(np.sum(x*x))
				}

	if weight:
		rank_types = ['weight']
	else:
		rank_types = ['act','grad','actxgrad']

	for rank_type in rank_types:
		for layer in range(params['num_layers']):
			col = df.loc[df['layer']==layer][rank_type+'_rank']
			norm_constant = norm_funcs[norm](col)
			if norm_constant == 0:
				print('norm constant value 0 for rank type %s and layer %s'%(rank_type,str(layer)))
			else:
				df[rank_type+'_rank'] = np.where(df['layer'] == layer ,df[rank_type+'_rank']/norm_constant,df[rank_type+'_rank'] )
	
	return df
		
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


#MISC FORMATTING FUNCTIONS

def nodeid_2_perlayerid(nodeid,params):    #takes in node unique id outputs tuple of layer and within layer id
	imgnode_names = params['imgnode_names']
	layer_nodes = params['layer_nodes']
	if isinstance(nodeid,str):
		if not nodeid.isnumeric():
			layer = 'img'
			layer_name='img'
			within_layer_id = imgnode_names.index(nodeid)
			return layer,within_layer_id, layer_name
	nodeid = int(nodeid)
	total= 0
	for i in range(len(layer_nodes)):
		total += len(layer_nodes[i][1])
		if total > nodeid:
			layer = i
			layer_name = layer_nodes[i][0]
			within_layer_id = layer_nodes[i][1].index(nodeid)
			break
	#layer = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['layer'].item()
	#within_layer_id = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['node_num_by_layer'].item()
	return layer,within_layer_id,layer_name

def layernum2name(layer,offset=1,title = 'layer'):
	return title+' '+str(layer+offset)

def get_nth_element_from_nested_list(l,n):    #this seems to come up with the nested layer lists
	flat_list = [item for sublist in l for item in sublist]
	return flat_list[n]
  

def layer_2_dissected_conv2d(target_layer,module, index=0, found=None):
	for layer, (name, submodule) in enumerate(module._modules.items()):
		if isinstance(submodule, dissected_Conv2d):
			if index==target_layer:
				found = submodule
			index+=1
		elif len(list(submodule.children())) > 0:
			found, index = layer_2_dissected_conv2d(target_layer,submodule, index=index, found=found)
	return found, index


def get_activations_from_dissected_Conv2d_modules(module,layer_activations=None):     
	if layer_activations is None:    #initialize the output dictionary if we are not recursing and havent done so yet
		layer_activations = {'nodes':[],'edges_in':[],'edges_out':[]}
	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, dissected_Conv2d):
			layer_activations['nodes'].append(submodule.postbias_out.cpu().detach().numpy())
			layer_activations['edges_in'].append(submodule.input.cpu().detach().numpy())
			layer_activations['edges_out'].append(submodule.format_edges(data= 'activations'))
			#print(layer_activations['edges_out'][-1].shape)
		elif len(list(submodule.children())) > 0:
			layer_activations = get_activations_from_dissected_Conv2d_modules(submodule,layer_activations=layer_activations)   #module has modules inside it, so recurse on this module

	return layer_activations

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

def rgb2hex(r, g, b):
	return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def image2plot(image_path,layout,resize = False,size = (32,32)):
	img = Image.open(image_path)
	if resize:
		img = img.resize(size,resample=Image.NEAREST)

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

#this is currently unused as edge_inputs are used for each channel image
def get_channelwise_image(image_name,channel,input_image_directory):    
	#THIS NEEDS TO BE NORMALIZED AS PER THE MODELS DATALOADER
	im = Image.open(input_image_directory+image_name)
	np_full_im = np.array(im)
	return np_full_im[:,:,channel]

def preprocess_image(image_path,params):
	preprocess = params['preprocess']
	cuda = params['cuda']
	#image loading 
	image_name = image_path.split('/')[-1]
	image = Image.open(image_path)
	image = preprocess(image).float()
	image = image.unsqueeze(0)
	if cuda:
		image = image.cuda()
	return image


#NODE FUNCTIONS

def node_color_scaling(x):
	return x
	#return -(x-1)**4+1

def gen_node_colors(nodes_df,rank_type,params):
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
			alpha = node_color_scaling(node_weight)
			node_colors[-1].append(layer_colors[layer%len(layer_colors)]+str(round(alpha,3))+')')
			
	return node_colors,node_weights



#EDGE FUNCTIONS
# def load_category_rank_data(category,root_path,part='edges'):
#     ranks = torch.load(os.path.join(root_path,'categories_%s'%part,'%s_%s_rank.pt'%(category,part)))
#     dflist = []

#     if part == 'edges':
#         edge_num = 0
#         for layer in range(len(ranks['act'])):
#             for out_channel in range(len(ranks['act'][layer])):
#                 for in_channel in range(len(ranks['act'][layer][out_channel])):
#                     dflist.append([edge_num,layer,out_channel,in_channel,ranks['act'][layer][out_channel][in_channel],ranks['grad'][layer][out_channel][in_channel],ranks['weight'][layer][out_channel][in_channel],ranks['actxgrad'][layer][out_channel][in_channel],category])
#                     edge_num += 1
#         edge_column_names = ['edge_num','layer','out_channel','in_channel','act_rank','grad_rank','weight_rank','actxgrad_rank','category']
#         df = pd.DataFrame(dflist,columns=edge_column_names)

#     else:
#         node_num = 0
#         for layer in range(len(ranks['act'])):
#             for num_by_layer in range(len(ranks['act'][layer])):
#                 dflist.append([node_num,layer,num_by_layer,ranks['act'][layer][num_by_layer],ranks['grad'][layer][num_by_layer],ranks['weight'][layer][num_by_layer],ranks['actxgrad'][layer][num_by_layer],category])
#                 node_num += 1

#         #make nodes DF
#         node_column_names = ['node_num','layer','node_num_by_layer','act_rank','grad_rank','weight_rank','actxgrad_rank','category']
#         df = pd.DataFrame(allnode_dflist,columns=node_column_names)

#     return df

def edge_width_scaling(x):
	#return max(.4,(x*10)**1.7)
	return max(.4,np.exp(2.5*x))

def edge_color_scaling(x):
	#return max(.7,-(x-1)**4+1)
	return max(.7, x)


def get_thresholded_edges(threshold,rank_type,df,target_category):          #just get those edges that pass the threshold criteria for the target category
	if len(threshold) != 2:
		raise Exception('length of threshold needs to be two ([lower, higher])')
	return df.loc[(df[rank_type+'_rank'] >= threshold[0]) & (df[rank_type+'_rank'] <= threshold[1])]

def get_max_edge_widths(edge_widths):
	maxes = []
	for layer in range(len(edge_widths)):
		if len(edge_widths[layer]) >0:
			maxes.append(edge_widths[layer].index(max(edge_widths[layer])))
		else:
			maxes.append(None)
	return maxes

def gen_edge_graphdata(df, node_positions, rank_type, target_category, params, num_hoverpoints=15):
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
		#color
		alpha = edge_color_scaling(row[edges_df_columns.index(rank_type+'_rank')+1])
		colors[row.layer].append(layer_colors[row.layer%len(layer_colors)]+str(round(alpha,3))+')')
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

def check_edge_validity(nodestring,params):
	from_node = nodestring.split('-')[0]
	to_node = nodestring.split('-')[1]
	try:
		from_layer,from_within_id,from_layer_name = nodeid_2_perlayerid(from_node,params)
		to_layer,to_within_id,to_layer_name = nodeid_2_perlayerid(to_node,params)
		#check for valid edge
		valid_edge = False
		if from_layer=='img':
			if to_layer== 0:
				valid_edge = True
		elif to_layer == from_layer+1:
			valid_edge = True
		if not valid_edge:
			print('invalid edge name')
			return [False, None, None, None, None]
		return True, from_layer,to_layer,from_within_id,to_within_id
	except:
		#print('exception')
		return [False, None, None, None, None] 

def edgename_2_singlenum(model,edgename,params):
	valid, from_layer,to_layer,from_within_id,to_within_id = check_edge_validity(edgename,params)
	if not valid:
		raise ValueError('edgename %s is invalid'%edgename)
	conv_module = layer_2_dissected_conv2d(int(to_layer),model)[0]
	return conv_module.add_indices[int(to_within_id)][int(from_within_id)]

def edgename_2_edge_figures(edgename, image_name, kernels, activations, params):  #returns truth value of valid edge and kernel if valid
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

#ACTIVATION MAP FUNCTIONS

#run through all model modules recursively, and pull the activations stored in dissected_Conv2d modules 
def get_activations_from_dissected_Conv2d_modules(module,layer_activations=None):     
	if layer_activations is None:    #initialize the output dictionary if we are not recursing and havent done so yet
		layer_activations = {'nodes':[],'edges_in':[],'edges_out':[]}
	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, dissected_Conv2d):
			layer_activations['nodes'].append(submodule.postbias_out.cpu().detach().numpy())
			layer_activations['edges_in'].append(submodule.input.cpu().detach().numpy())
			layer_activations['edges_out'].append(submodule.format_edges(data= 'activations'))
			#print(layer_activations['edges_out'][-1].shape)
		elif len(list(submodule.children())) > 0:
			layer_activations = get_activations_from_dissected_Conv2d_modules(submodule,layer_activations=layer_activations)   #module has modules inside it, so recurse on this module

	return layer_activations

#reformat activations so images dont take up a dimension in the np array, 
# but rather there is an individual array for each image name key in a dict
def act_array_2_imgname_dict(layer_activations, image_names):
	new_activations = {'nodes':{},'edges_in':{},'edges_out':{}}
	for i in range(len(image_names)):
		for part in ['nodes','edges_in','edges_out']:
			new_activations[part][image_names[i]] = []
			for l in range(len(layer_activations[part])):
				new_activations[part][image_names[i]].append(layer_activations[part][l][i])
	return new_activations

def get_model_activations_from_image(image_path, model_dis, params):
	cuda = params['cuda']
	model_dis = set_across_model(model_dis,'target_node',None)
	#image loading 
	image = preprocess_image(image_path,params)
	image_name = image_path.split('/')[-1]
	#pass image through model
	output = model_dis(image)
	#recursively fectch activations in conv2d_dissected modules
	layer_activations = get_activations_from_dissected_Conv2d_modules(model_dis)
	#return correctly formatted activation dict
	return act_array_2_imgname_dict(layer_activations,[image_name])

def combine_activation_dicts(all_activations,new_activations):       #when you get activations for a new image add those image keys to your full activation dict
	for key in ['nodes','edges_in','edges_out']:
		all_activations[key].update(new_activations[key])
	return all_activations

# def update_all_activations(image_path,model_dis,params):
# 	image_name = image_path.split('/')[-1]
# 	print('dont have activations for %s in memory, fetching by running model'%image_name)
# 	global all_activations
# 	new_activations = get_model_activations_from_image(image_path, model_dis, params)
# 	all_activations = combine_activation_dicts(all_activations,new_activations)
	
# 	if params['dynamic_input']:
# 		global activations_cache_order
# 		activations_cache_order.append(image_name)
# 		if len(activations_cache_order) > params['dynamic_act_cache_num']:
# 			for key in ['nodes','edges_in','edges_out']:
# 				del all_activations[key][activations_cache_order[0]]
# 			del activations_cache_order[0]

#RANK FUNCTIONS

def get_ranks_from_dissected_Conv2d_modules(module,layer_ranks=None,layer_normalizations=None,weight_rank=False):     #run through all model modules recursively, and pull the ranks stored in dissected_Conv2d modules 
	if layer_ranks is None:    #initialize the output dictionary if we are not recursing and havent done so yet
		if weight_rank:
			layer_ranks = {'nodes':{'weight':[]},'edges':{'weight':[]}}
		else:
			layer_ranks = {'nodes':{'act':[],'grad':[],'actxgrad':[]},
						   'edges':{'act':[],'grad':[],'actxgrad':[]}}

	for layer, (name, submodule) in enumerate(module._modules.items()):
		#print(submodule)
		if isinstance(submodule, dissected_Conv2d):
			submodule.average_ranks()
			if weight_rank:
				rank_types = ['weight']
			else:
				rank_types = ['act','grad','actxgrad']

			for rank_type in rank_types:
				#submodule.gen_normalizations(rank_type)
				layer_ranks['nodes'][rank_type].append([submodule.name,submodule.postbias_ranks[rank_type].cpu().detach().numpy()])
				layer_ranks['edges'][rank_type].append([submodule.name,submodule.format_edges(data= 'ranks',rank_type=rank_type,weight_rank=weight_rank)])
				#print(layer_ranks['edges'][-1].shape)
		elif len(list(submodule.children())) > 0:
			layer_ranks = get_ranks_from_dissected_Conv2d_modules(submodule,layer_ranks=layer_ranks,weight_rank=weight_rank)   #module has modules inside it, so recurse on this module
	return layer_ranks



def get_model_ranks_for_category(category, target_node, model_dis,params):

	device = torch.device("cuda" if params['cuda'] else "cpu")
	criterion = params['criterion']
	####SET UP MODEL
	model_dis = set_across_model(model_dis,'target_node',None)
	if target_node is not 'loss':
		target_node_layer,target_node_within_layer_id,target_node_layer_name = nodeid_2_perlayerid(target_node,params)
		model_dis=set_model_target_node(model_dis,target_node_layer,target_node_within_layer_id)

	model_dis = set_across_model(model_dis,'clear_ranks',False)

	node_ranks = {}
	edge_ranks = {}


	####SET UP DATALOADER
	kwargs = {'num_workers': params['num_workers'], 'pin_memory': True} if params['cuda'] else {}

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
					criterion(output, Variable(target)).backward()    #running backward pass calls all the hooks and calculates the ranks of all edges and nodes in the graph 
			except TargetReached:
				print('target node %s reached, halted forward pass'%str(target_node))

			#	torch.sum(output).backward()    # run backward pass with respect to net outputs rather than loss function

	layer_ranks = get_ranks_from_dissected_Conv2d_modules(model_dis)

	model_dis = set_across_model(model_dis,'clear_ranks',True)

	return layer_ranks

def get_model_ranks_from_image(image_path, target_node, model_dis, params): 
	#model_dis.clear_ranks_func()  #so ranks dont accumulate
	cuda = params['cuda']
	device = torch.device("cuda" if cuda else "cpu")
	criterion = params['criterion']
	#image loading 
	image_name = image_path.split('/')[-1]
	image,target = single_image_loader(image_path, params['preprocess'], label_file_path = params['label_file_path'])
	image, target = image.to(device), target.to(device)

	model_dis = set_across_model(model_dis,'target_node',None)
	if target_node != 'loss':
		target_node_layer,target_node_within_layer_id,target_node_layer_name = nodeid_2_perlayerid(target_node,params)
		model_dis=set_model_target_node(model_dis,target_node_layer,target_node_within_layer_id)


	#pass image through model
	try:
		output = model_dis(image)    #running forward pass sets up hooks and stores activations in each dissected_Conv2d module
		if target_node == 'loss':
			target = max_likelihood_for_no_target(target,output) 
			criterion(output, Variable(target)).backward()    #running backward pass calls all the hooks and calculates the ranks of all edges and nodes in the graph 
	except TargetReached:
		print('target node %s reached, halted forward pass'%str(target_node))

	layer_ranks = get_ranks_from_dissected_Conv2d_modules(model_dis)
	return layer_ranks



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
				   text=layer_nodes[layer][1],
				   hoverinfo='text'
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
