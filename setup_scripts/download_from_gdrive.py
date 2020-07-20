#quick script for downloading large files stored on google drive

from google_drive_downloader import GoogleDriveDownloader as gdd
from subprocess import call
import os

if not os.path.exists('../image_data'):
	os.mkdir('../image_data')
if not os.path.exists('../prepped_models'):
	os.mkdir('../prepped_models')


def tar_download(id,dest_path):
	print('downloading')
	gdd.download_file_from_google_drive(file_id=id, dest_path=dest_path, overwrite=True)
	print('untaring')
	out_dir = '/'.join(dest_path.split('/')[:-1])
	call('tar -xzvf %s -C %s'%(dest_path,out_dir),shell=True)
	call('rm %s'%dest_path,shell=True)	


print('LARGE FOLDER DOWNLOAD\n')
print('DOWNLOADING IMAGE-DATA FROM GDRIVE\n')

#cifar10
print('cifar10')
tar_download('17pjtPG-MJK7mhTh_KHvHLHUwSButkwLA','../image_data/cifar10.tgz')

#mnist
print('mnist')
tar_download('1KgGNthhon5og6ggdYhgQOi0zoB39RAF0','../image_data/mnist.tgz')


print('DOWNLOADING PREPPED-MODELS FROM GDRIVE\n')

#cifar10_prunned
print('cifar10_prunned')
tar_download('1GY-u1JC2PQaiXznHQ1nkV6lMDI0laJ7G','../prepped_models/cifar10_prunned.tgz')

#mnist
print('mnist')
tar_download('1WbiMi0JZ3XegtSTB0er8ShezaZDQCG2p','../prepped_models/mnist.tgz')

