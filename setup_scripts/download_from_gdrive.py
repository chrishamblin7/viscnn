#quick script for downloading large files stored on google drive

import requests
from subprocess import call
import os

online_models = {'mnist':{'prepped_model':'1p7ZjoUeexiu3-Fv2wCJT4sTaWVVrtCDD','model':'1X6wR6nJ_SguVzd6MVFelvXsH9G2uR4WZ','images':'1hHrA8ASShRz_JqDu48rQ5O7uWBwiXZ9X'},
				 'cifar10':{'prepped_model':'1GY-u1JC2PQaiXznHQ1nkV6lMDI0laJ7G','model':None,'images':'17pjtPG-MJK7mhTh_KHvHLHUwSButkwLA'}
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

	if not os.path.exists('../image_data'):
		os.mkdir('../image_data')
	if not os.path.exists('../prepped_models'):
		os.mkdir('../prepped_models')
	if not os.path.exists('../models'):
		os.mkdir('../models')

	if not args.only_download_images:
		print('Downloading prepped_model: %s\n\n'%args.model)
		tar_download(online_models[args.model]['prepped_model'],'../prepped_models/%s.tgz'%args.model)
		if online_models[args.model]['model'] is not None:
			print('Downloading model')
			file_download(online_models[args.model]['model'],'../models/%s_statedict.pt'%args.model)

	if not args.dont_download_images:
		print('Downloading input image data associated with: %s\n\n'%args.model)
		tar_download(online_models[args.model]['images'],'../image_data/%s.tgz'%args.model)