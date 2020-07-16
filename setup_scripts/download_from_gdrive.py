#quick script for downloading large files stored on google drive

from google_drive_downloader import GoogleDriveDownloader as gdd
from subprocess import call
import os

if not os.path.exists('../image_data'):
	os.mkdir('../image_data')
if not os.path.exists('../prepped_data'):
	os.mkdir('../prepped_models')


try:
	print('LARGE FOLDER DOWNLOAD\n')
	print('DOWNLOADING IMAGE-DATA FROM GDRIVE\n')
	#cifar10
	print('cifar10')
	print('downloading')
	gdd.download_file_from_google_drive(file_id='17pjtPG-MJK7mhTh_KHvHLHUwSButkwLA', dest_path='../image_data/cifar10.tgz', overwrite=True)
	print('untaring')
	call('tar -xzvf ../image_data/cifar10.tgz',shell=True)
	call('rm ../image_data/cifar10.tgz',shell=True)

	print('DOWNLOADING PREPPED-MODELS FROM GDRIVE\n')
	#cifar10_prunned
	print('cifar10_prunned')
	print('downloading')
	gdd.download_file_from_google_drive(file_id='1GY-u1JC2PQaiXznHQ1nkV6lMDI0laJ7G', dest_path='../prepped_models/cifar10_prunned.tgz', overwrite=True)
	print('untaring')
	call('tar -xzvf ../prepped_models/cifar10_prunned.tgz',shell=True)
	call('rm ../prepped_models/cifar10_prunned.tgz',shell=True)

except:
	print('hmmm something went wrong downloading the google drive files. You can try downloading them manually through the following urls:\n')
	print('https://drive.google.com/file/d/1QlVe2_uJlHNVwYBiUr1napQtlhkJdG2_/view?usp=sharing')
	print('https://drive.google.com/file/d/1gQrMwjlIcIHleWFlt3Sb6mjcvD42wALY/view?usp=sharing')
	print('\nFrom each link you should be able to download "image_data.tgz" and "prepped_models.tgz" respectively. Untar each (tar -xvzf image_data.tgz),' +
		  'and then place the resulting folders ("image_data" and "prepped_models") in the top folder of this repo.')