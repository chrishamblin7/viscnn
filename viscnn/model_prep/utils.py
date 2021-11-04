import torch
import os
import pickle
from viscnn.utils import update_sys_path

def load_prepped_model(prepped_model,device=None,dont_download_images=False):
	'''
	prepped_model: just a string of a folder within the 'prepped_models' folder
	device: if None, defaults to the device in 'prepped_model_parameters_used'
			else, use string like 'cuda:0', 'cuda:1', 'cpu' etc.
	'''
	#figure out where the prepped model is
	if '/' in prepped_model:
		prepped_model_path = os.path.abspath(prepped_model)
		prepped_model_folder = prepped_model.split('/')[-1]
	else:
		from viscnn import prepped_models_root_path
		prepped_model_path = prepped_models_root_path + '/' + prepped_model
		prepped_model_folder = prepped_model_path.split('/')[-1]
		
	if not os.path.isdir(prepped_model_path):
		#try to download prepped_model from gdrive
		from viscnn.download_from_gdrive import download_from_gdrive
		download_from_gdrive(prepped_model_folder,dont_download_images = dont_download_images)

	
	#load Model
	update_sys_path(prepped_model_path)
	import prep_model_params_used as prep_model_params
	model = prep_model_params.model

	if device is None:
		device = prep_model_params.device

	_ = model.to(device).eval()

	return model

def load_prepped_model_params(prepped_model,device=None,deepviz_neuron=None,deepviz_edge=False,dont_download_images=False):

	'''
	prepped_model: just a string of a folder within the 'prepped_models' folder
	'''
	
	#figure out where the prepped model is
	if '/' in prepped_model:
		prepped_model_path = os.path.abspath(prepped_model)
		prepped_model_folder = prepped_model.split('/')[-1]
	else:
		from viscnn import prepped_models_root_path
		prepped_model_path = prepped_models_root_path + '/' + prepped_model
		prepped_model_folder = prepped_model_path.split('/')[-1]
		
	if not os.path.isdir(prepped_model_path):
		#try to download prepped_model from gdrive
		from viscnn.download_from_gdrive import download_from_gdrive
		download_from_gdrive(prepped_model_folder,dont_download_images = dont_download_images)


	prepped_model_folder = prepped_model_path.split('/')[-1]

	params = {}
	params['prepped_model'] = prepped_model_folder
	params['prepped_model_path'] = prepped_model_path

	update_sys_path(prepped_model_path)
	import prep_model_params_used as prep_model_params

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
	if device is None:
		params['device'] = prep_model_params.device
	else:
		params['device'] = device
	params['input_image_directory'] = prep_model_params.input_img_path+'/'
	params['preprocess'] = prep_model_params.preprocess     #torchvision transfrom to pass input images through
	params['label_file_path'] = prep_model_params.label_file_path
	params['criterion'] = prep_model_params.criterion
	params['rank_img_path'] = prep_model_params.rank_img_path
	params['num_workers'] = prep_model_params.num_workers
	params['seed'] = prep_model_params.seed
	params['batch_size'] = prep_model_params.batch_size

	#misc graph data
	misc_data = pickle.load(open(prepped_model_path+'/misc_graph_data.pkl','rb'))
	params['layer_nodes'] = misc_data['layer_nodes']
	params['num_layers'] = misc_data['num_layers']
	params['num_nodes'] = misc_data['num_nodes']
	params['categories'] = misc_data['categories']
	params['num_img_chan'] = misc_data['num_img_chan']
	params['imgnode_positions'] = misc_data['imgnode_positions']
	params['imgnode_colors'] = misc_data['imgnode_colors']
	params['imgnode_names'] = misc_data['imgnode_names']
	params['ranks_data_path'] = prepped_model_path+'/ranks/'
	
	#input images
	params['input_image_directory'] = prep_model_params.input_img_path+'/'
	params['input_image_list'] = os.listdir(params['input_image_directory'])
	params['input_image_list'].sort()

	return params