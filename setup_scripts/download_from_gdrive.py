#quick script for downloading large files stored on google drive

from google_drive_downloader import GoogleDriveDownloader as gdd
from subprocess import call
import os

online_models = {'mnist':{'prepped_model':'1p7ZjoUeexiu3-Fv2wCJT4sTaWVVrtCDD','model':'1AjjInlHlo-dEq4NKcrCANXcKDJ8xocn7','images':'1hHrA8ASShRz_JqDu48rQ5O7uWBwiXZ9X'},
				 'cifar10':{'prepped_model':'1GY-u1JC2PQaiXznHQ1nkV6lMDI0laJ7G','model':None,'images':'17pjtPG-MJK7mhTh_KHvHLHUwSButkwLA'}
				}

online_model_names = list(online_models.keys())

def tar_download(id,dest_path):
	print('downloading')
	gdd.download_file_from_google_drive(file_id=id, dest_path=dest_path, overwrite=True)
	print('untaring')
	out_dir = '/'.join(dest_path.split('/')[:-1])
	call('tar -xzvf %s -C %s'%(dest_path,out_dir),shell=True)
	call('rm %s'%dest_path,shell=True)	

def file_download(id,dest_path):
	print('downloading')
	gdd.download_file_from_google_drive(file_id=id, dest_path=dest_path, overwrite=True)

if __name__ == "__main__":

	import argeparse

	def get_args():
		parser = argparse.ArgumentParser()
		parser.add_argument("model", type = str, choices=online_model_names,
							help="Name of model (the folder names within 'prepped_models'). Can also be model avaiable online.")
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


# print('LARGE FOLDER DOWNLOAD\n')
# print('DOWNLOADING IMAGE-DATA FROM GDRIVE\n')

# #cifar10
# print('cifar10')
# tar_download('17pjtPG-MJK7mhTh_KHvHLHUwSButkwLA','../image_data/cifar10.tgz')

# #mnist
# print('mnist')
# tar_download('1KgGNthhon5og6ggdYhgQOi0zoB39RAF0','../image_data/mnist.tgz')


# print('DOWNLOADING PREPPED-MODELS FROM GDRIVE\n')

# #cifar10_prunned
# print('cifar10_prunned')
# tar_download('1GY-u1JC2PQaiXznHQ1nkV6lMDI0laJ7G','../prepped_models/cifar10_prunned.tgz')

# #mnist
# print('mnist')
# tar_download('1WbiMi0JZ3XegtSTB0er8ShezaZDQCG2p','../prepped_models/mnist.tgz')


