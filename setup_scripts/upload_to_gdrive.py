import os
from subprocess import call
import argparse


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("folder_path", type = str,
						help="relative path to folder to upload (dont use trailing '/' !)")
	parser.add_argument('upload_folder', type = str,
						help="folder name to upload to on gdrive (should be 'image_data','models', or 'prepped_models')")
	parser.add_argument('--file', action='store_true', help='upload a file, so no tarball')
	args = parser.parse_args()
	return args

args = get_args()

split_folder_path = args.folder_path.split('/')
root_path = '/'.join(split_folder_path[:-1])
folder = split_folder_path[-1]
print('switching to directory %s'%root_path)
os.chdir(root_path)
if not args.file:
	print('taring folder %s'%folder)
	call("tar -czvf %s.tar.gz --exclude 'cache' --exclude '__pycache__' --exclude 'ranking_images' --exclude 'subgraphs' %s"%(folder,folder),shell=True)
	print('uploading %s tarball to gdrive folder %s'%(folder,args.upload_folder))
	call("gupload %s.tar.gz -c %s"%(folder,args.upload_folder),shell=True)
else:
	print('uploading %s to gdrive folder %s'%(folder,args.upload_folder))
	call("gupload %s -c %s"%(folder,args.upload_folder),shell=True)