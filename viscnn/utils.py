#MISC UTILITY FUNCTIONS
import sys
import os
from PIL import Image
import torch
from torch import nn
from viscnn.dissected_Conv2d import *


### IMAGE PROCESSING ###

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


def preprocess_image(image_path,params):
	preprocess = params['preprocess']

	#image loading 
	image_name = image_path.split('/')[-1]
	image = Image.open(image_path)
	image = preprocess(image).float()
	image = image.unsqueeze(0)
	image = image.to(params['device'])
	return image


#### NAMING  ####

#return list of names for conv modules based on their nested module names '_' seperated
def get_conv_full_names(model,mod_names = [],mod_full_names = []):
    #gen names based on nested modules
    for name, module in model._modules.items():
        if len(list(module.children())) > 0:
            mod_names.append(str(name))
            # recurse
            mod_full_names = get_conv_full_names(module,mod_names = mod_names, mod_full_names = mod_full_names)
            mod_names.pop()

        if isinstance(module, torch.nn.modules.conv.Conv2d):    # found a 2d conv module
            mod_full_names.append('_'.join(mod_names+[name]))
            #new_module = dissected_Conv2d(module, name='_'.join(mod_names+[name]), store_activations=store_activations,store_ranks=store_ranks,clear_ranks=clear_ranks,cuda=cuda,device=device) 
            #model._modules[name] = new_module
    return mod_full_names     

    
#return list of names for conv modules based on their simple order, first conv is 'conv1', then 'conv2' etc. 
def get_conv_simple_names(model):
    names = []
    count = 0
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            names.append('conv'+str(count))
            count+=1
    return names
 
# returns a dict that maps simple names to full names
def gen_conv_name_dict(model):
    simple_names = get_conv_simple_names(model)
    full_names = get_conv_full_names(model)
    return dict(zip(simple_names, full_names))


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


### TENSORS ###

def unravel_index(indices,shape):
    r"""Converts flat indices into unraveled coordinates in a target shape.

    This is a `torch` implementation of `numpy.unravel_index`.

    Args:
        indices: A tensor of (flat) indices, (*, N).
        shape: The targeted shape, (D,).

    Returns:
        The unraveled coordinates, (*, N, D).
    """

    coord = []

    for dim in reversed(shape):
        coord.append(indices % dim)
        indices = indices // dim

    coord = torch.stack(coord[::-1], dim=-1)

    return coord


###  NETWORKS ###

def relu(array):
	neg_indices = array < 0
	array[neg_indices] = 0
	return array


### COLOR

def rgb2hex(r, g, b):
	return '#{:02x}{:02x}{:02x}'.format(r, g, b)

def color_vec_2_str(colorvec,a='1'):
    return 'rgba(%s,%s,%s,%s)'%(str(int(colorvec[0])),str(int(colorvec[1])),str(int(colorvec[2])),a)


### TRULY MISC ###

def get_nth_element_from_nested_list(l,n):    #this seems to come up with the nested layer lists
	flat_list = [item for sublist in l for item in sublist]
	return flat_list[n]

def update_sys_path(path):
	full_path = os.path.abspath(path)
	if full_path not in sys.path:
		sys.path.insert(0,full_path)

def minmax_normalize_between_values(vec,min_v,max_v):
    return (max_v-min_v)*(vec-np.min(vec))/(np.max(vec)-np.min(vec))+min_v
    
def min_distance(x,y,minimum=1):
    dist = np.linalg.norm(x-y)
    if dist > minimum:
        return dist,True
    else:
        return dist,False
    
def multipoint_min_distance(points):   #takes numpy array of shape (# points, # dimensions)
    dist_mat = distance_matrix(points,points)
    dist_mat[np.tril_indices(dist_mat.shape[0], 0)] = 10000
    print(dist_mat)
    return np.min(dist_mat)

