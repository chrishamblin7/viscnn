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



#this is currently unused as edge_inputs are used for each channel image
def get_channelwise_image(image_name,channel,input_image_directory):    
	#THIS NEEDS TO BE NORMALIZED AS PER THE MODELS DATALOADER
	im = Image.open(input_image_directory+image_name)
	np_full_im = np.array(im)
	return np_full_im[:,:,channel]

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