from utils import preprocess_image


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
	print('running model to fetch activations')
	cuda = params['cuda']
	model_dis = set_across_model(model_dis,'target_node',None)
	if 'device' in params.keys():
		model_dis.to(params['device'])
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

