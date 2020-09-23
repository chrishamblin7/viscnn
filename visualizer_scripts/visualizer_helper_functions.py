#various functions used by the visualizer tool
import os
from copy import deepcopy
import torch
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import sys
sys.path.insert(0, os.path.abspath('../prep_model_scripts/'))
from dissected_Conv2d import *

#DATAFRAME FUNCTIONS

def rank_file_2_df(file_path):      #takes a node or edge 'rank.pt' file and turns it into a pandas dataframe
    ranks = torch.load(file_path)
    
    #nodes
    if 'node' in file_path.split('/')[-1]:
        node_dflist = []
        node_num = 0
        for layer in range(len(ranks['act'])):
            for num_by_layer in range(len(ranks['act'][layer])):
                node_dflist.append([node_num,layer,num_by_layer,ranks['act'][layer][num_by_layer],ranks['grad'][layer][num_by_layer],ranks['weight'][layer][num_by_layer],ranks['actxgrad'][layer][num_by_layer]])
                node_num += 1
        #make nodes DF
        node_column_names = ['node_num','layer','node_num_by_layer','act_rank','grad_rank','weight_rank','actxgrad_rank']
        df = pd.DataFrame(node_dflist,columns=node_column_names)

    elif 'edge' in file_path.split('/')[-1]:
        #edges
        edge_dflist = []
        edge_num = 0
        for layer in range(len(ranks['act'])):
            for out_channel in range(len(ranks['act'][layer])):
                for in_channel in range(len(ranks['act'][layer][out_channel])):
                    edge_dflist.append([edge_num,layer,out_channel,in_channel,ranks['act'][layer][out_channel][in_channel],ranks['grad'][layer][out_channel][in_channel],ranks['weight'][layer][out_channel][in_channel],ranks['actxgrad'][layer][out_channel][in_channel]])
                    edge_num += 1
        edge_column_names = ['edge_num','layer','out_channel','in_channel','act_rank','grad_rank','weight_rank','actxgrad_rank']
        df = pd.DataFrame(edge_dflist,columns=edge_column_names)
    
    else:
        raise Exception("Can't determine if %s is node or edge rank. Make sure 'node' or 'edge' is in file name"%file_path)

    return df


#MISC FORMATTING FUNCTIONS

def nodeid_2_perlayerid(nodeid,nodes_df,params):    #takes in node unique id outputs tuple of layer and within layer id
    imgnode_names = params['imgnode_names']
    layer_nodes = params['layer_nodes']
    if isinstance(nodeid,str):
        if not nodeid.isnumeric():
            layer = 'img'
            within_layer_id = imgnode_names.index(nodeid)
            return layer,within_layer_id
    nodeid = int(nodeid)
    layer = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['layer'].item()
    within_layer_id = nodes_df[nodes_df['category']=='overall'][nodes_df['node_num'] == nodeid]['node_num_by_layer'].item()
    return layer,within_layer_id

def layernum2name(layer,offset=1,title = 'layer'):
    return title+' '+str(layer+offset)

def get_nth_element_from_nested_list(l,n):    #this seems to come up with the nested layer lists
    flat_list = [item for sublist in l for item in sublist]
    return flat_list[n]
  

#INPUT IMAGE FUNCTIONS

def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


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


#NODE FUNCTIONS

def node_color_scaling(x):
    return -(x-1)**4+1

def gen_node_colors(target_category,rank_type,nodes_df,params):
    layer_nodes = params['layer_nodes']
    layer_colors = params['layer_colors']

    node_colors = []
    node_weights = []
    for layer in layer_nodes:
        node_colors.append([])
        node_weights.append([])
        for node in layer_nodes[layer]:
            node_weight = nodes_df[nodes_df['category']==target_category].iloc[node][rank_type+'_rank']
            node_weights[-1].append(node_weight)
            alpha = node_color_scaling(node_weight)
            node_colors[-1].append(layer_colors[layer%len(layer_colors)]+str(round(alpha,3))+')')
            
    return node_colors,node_weights



#EDGE FUNCTIONS
def load_category_edge_data(category,root_path):
    dflist = []
    ranks = torch.load(os.path.join(root_path,'%s_edges_rank.pt'%category))
    edge_num = 0
    for layer in range(len(ranks['act'])):
        for out_channel in range(len(ranks['act'][layer])):
            for in_channel in range(len(ranks['act'][layer][out_channel])):
                dflist.append([edge_num,layer,out_channel,in_channel,ranks['act'][layer][out_channel][in_channel],ranks['grad'][layer][out_channel][in_channel],ranks['weight'][layer][out_channel][in_channel],ranks['actxgrad'][layer][out_channel][in_channel],category])
                edge_num += 1

    edge_column_names = ['edge_num','layer','out_channel','in_channel','act_rank','grad_rank','weight_rank','actxgrad_rank','category']
    edges_df = pd.DataFrame(dflist,columns=edge_column_names)
    return edges_df

def edge_width_scaling(x):
    return max(.4,(x*10)**1.7)

def edge_color_scaling(x):
    return max(.7,-(x-1)**4+1)


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

    edges_df_columns = ['edge_num', 'layer', 'out_channel', 'in_channel', 'act_rank',
       'grad_rank', 'weight_rank', 'actxgrad_rank', 'category']
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
        out_node = layer_nodes[row.layer][row.out_channel]
        if row.layer != 0:
            in_node = layer_nodes[row.layer-1][row.in_channel]
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

def check_edge_validity(nodestring,nodes_df,params):
    from_node = nodestring.split('-')[0]
    to_node = nodestring.split('-')[1]
    try:
        from_layer,from_within_id = nodeid_2_perlayerid(from_node,nodes_df,params)
        to_layer,to_within_id = nodeid_2_perlayerid(to_node,nodes_df,params)
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

def edgename_2_edge_figures(edgename, image_name, kernels, activations, nodes_df, params):  #returns truth value of valid edge and kernel if valid
    valid,from_layer,to_layer,from_within_id,to_within_id  = check_edge_validity(edgename,nodes_df,params)
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
			print(layer_activations['edges_out'][-1].shape)
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
    preprocess = params['preprocess']
    cuda = params['cuda']

    #image loading 
    image_name = image_path.split('/')[-1]
    image = Image.open(image_path)
    image = preprocess(image).float()
    image = image.unsqueeze(0)
    if cuda:
        image = image.cuda()
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


def update_all_activations(image_path,model_dis,params):
    image_name = image_path.split('/')[-1]
    print('dont have activations for %s in memory, fetching by running model'%image_name)
    global all_activations
    new_activations = get_model_activations_from_image(image_path, model_dis, params)
    all_activations = combine_activation_dicts(all_activations,new_activations)
    
    if params['dynamic_input']:
        global activations_cache_order
        activations_cache_order.append(image_name)
        if len(activations_cache_order) > params['dynamic_act_cache_num']:
            for key in ['nodes','edges_in','edges_out']:
                del all_activations[key][activations_cache_order[0]]
            del activations_cache_order[0]


#NETWORK GRAPH FUNCTIONS

def gen_networkgraph_traces(state,params,nodes_df):
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
        select_layer,select_position = nodeid_2_perlayerid(state['node_select_history'][-1],nodes_df,params)
    for layer in layer_nodes:
        #add nodes
        colors = deepcopy(state['node_colors'][layer])
        if layer == select_layer:
            colors[select_position] = 'rgba(0,0,0,1)'
        node_trace=go.Scatter3d(x=state['node_positions'][layer]['X'],
                   y=state['node_positions'][layer]['Y'],
                   z=state['node_positions'][layer]['Z'],
                   mode='markers',
                   name=layernum2name(layer,title = 'nodes'),
                   marker=dict(symbol='circle',
                                 size=6,
                                 opacity=.99,
                                 color=colors,
                                 #colorscale='Viridis',
                                 line=dict(color='rgb(50,50,50)', width=.5)
                                 ),
                   text=layer_nodes[layer],
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
                                    name=layernum2name(layer ,title = 'edges'),
                                    mode='lines',
                                    #line=dict(color=edge_colors_dict[layer], width=1.5),
                                    line=dict(color=color, width=state['edge_widths'][layer][edge_num]),
                                    text = state['edge_names'][layer][edge_num],
                                    hoverinfo='text'
                                    )
            edge_traces.append(edge_trace)


    combined_traces = imgnode_traces+node_traces+edge_traces
    return combined_traces
