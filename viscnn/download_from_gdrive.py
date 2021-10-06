#quick script for downloading large files stored on google drive

import requests
from subprocess import call
import os
from viscnn import root_path

#Dictionary with the online gdrive keys for models image fodler etc.
online_models = {'mnist':{'prepped_model':'1rRqywhDNIngaOI7wjBNg8ob041k9I1RC','model':'1X6wR6nJ_SguVzd6MVFelvXsH9G2uR4WZ','images':'1rGXi_pWGvz3UsdO1FpkWc2WsU-M42Z3v'},
				 'cifar10':{'prepped_model':'1GY-u1JC2PQaiXznHQ1nkV6lMDI0laJ7G','model':None,'images':'17pjtPG-MJK7mhTh_KHvHLHUwSButkwLA'},
				 'alexnet':{'prepped_model':'1Xpirw_Ss_wfOtukJRZDG1VUtVeSAdQpO','model':None,'images':'1NRbJJebFnyuqezFqMQ1qY53w5mXmelEl'},
				 'alexnet_sparse':{'prepped_model':'1023KCGDFeyYjdB8KOFN_mc57J45Z7yzM','model':'1MMr2LgwQkQIDb8SNwqLaqnezXJOVUHps','images':'1NRbJJebFnyuqezFqMQ1qY53w5mXmelEl'},
				}

online_model_names = list(online_models.keys())

def file_download(id, destination):
	URL = "https://docs.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params = { 'id' : id }, stream = True)
	token = get_confirm_token(response)

	if token:
		params = { 'id' : id, 'confirm' : token }
		response = session.get(URL, params = params, stream = True)

	save_response_content(response, destination)    

def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None

def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk: # filter out keep-alive new chunks
				f.write(chunk)


def tar_download(id,dest_path):
	print('downloading')
	file_download(id,dest_path)
	print('untaring')
	out_dir = '/'.join(dest_path.split('/')[:-1])
	call('tar -xzvf %s -C %s'%(dest_path,out_dir),shell=True)
	call('rm %s'%dest_path,shell=True)	


def download_from_gdrive(model,dont_download_images = False,only_download_images=False):
	#make image directories
	print('downloading %s from googledrive'%model)
	if not os.path.exists(root_path+'/image_data'):
		os.mkdir(root_path+'/image_data')
	if not os.path.exists(root_path+'/prepped_models'):
		os.mkdir(root_path+'/prepped_models')
	if not os.path.exists(root_path+'/models'):
		os.mkdir(root_path+'/models')

	if not only_download_images:
		print('Downloading prepped_model: %s\n\n'%model)
		tar_download(online_models[model]['prepped_model'],'../prepped_models/%s.tgz'%model)
		if not os.path.exists('../prepped_models/%s/subgraphs'%model):
			os.mkdir('../prepped_models/%s/subgraphs'%model)
			os.mkdir('../prepped_models/%s/subgraphs/info'%model)
			os.mkdir('../prepped_models/%s/subgraphs/models'%model)
			os.mkdir('../prepped_models/%s/subgraphs/visualizations'%model)
		if online_models[model]['model'] is not None:
			print('Downloading model')
			file_download(online_models[model]['model'],'../models/%s_statedict.pt'%model)

	if not dont_download_images:
		print('Downloading input image data associated with: %s\n\n'%model)
		tar_download(online_models[model]['images'],'../image_data/%s.tgz'%model)

if __name__ == "__main__":

	import argparse

	def get_args():
		parser = argparse.ArgumentParser()
		parser.add_argument("model", type = str, choices=online_model_names,
							help="Name of model (the folder names within 'prepped_models'). Can also be model available online.")
		parser.add_argument("--dont-download-images", action='store_true', 
							help='Dont download the image_data folder associated with that model.')
		parser.add_argument("--only-download-images", action='store_true', 
							help='dont download the actual prepped_model data you already have it, just download the associated image_data.')				
		args = parser.parse_args()
		return args

	args = get_args()

	download_from_gdrive(args.model,dont_download_images = args.dont_download_images,only_download_images=args.only_download_images)

	# if not os.path.exists('../image_data'):
	# 	os.mkdir('../image_data')
	# if not os.path.exists('../prepped_models'):
	# 	os.mkdir('../prepped_models')
	# if not os.path.exists('../models'):
	# 	os.mkdir('../models')

	# if not args.only_download_images:
	# 	print('Downloading prepped_model: %s\n\n'%args.model)
	# 	tar_download(online_models[args.model]['prepped_model'],'../prepped_models/%s.tgz'%args.model)
	# 	if not os.path.exists('../prepped_models/%s/subgraphs'%args.model):
	# 		os.mkdir('../prepped_models/%s/subgraphs'%args.model)
	# 		os.mkdir('../prepped_models/%s/subgraphs/info'%args.model)
	# 		os.mkdir('../prepped_models/%s/subgraphs/models'%args.model)
	# 		os.mkdir('../prepped_models/%s/subgraphs/visualizations'%args.model)
	# 	if online_models[args.model]['model'] is not None:
	# 		print('Downloading model')
	# 		file_download(online_models[args.model]['model'],'../models/%s_statedict.pt'%args.model)

	# if not args.dont_download_images:
	# 	print('Downloading input image data associated with: %s\n\n'%args.model)
	# 	tar_download(online_models[args.model]['images'],'../image_data/%s.tgz'%args.model)