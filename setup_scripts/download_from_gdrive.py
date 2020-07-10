#quick script for downloading large files stored on google drive

from google_drive_downloader import GoogleDriveDownloader as gdd
from subprocess import call

try:
	print('LARGE FOLDER DOWNLOAD\n')
	print('DOWNLOADING IMAGE-DATA FROM GDRIVE\n')
	gdd.download_file_from_google_drive(file_id='1QlVe2_uJlHNVwYBiUr1napQtlhkJdG2_', dest_path='../image_data.tgz', overwrite=True)
	print('\nUNTARING image_data.tgz\n')
	call('tar -xzvf ../image_data.tgz',shell=True)
	print('DOWNLOADING PREPPED-MODELS FROM GDRIVE\n')
	gdd.download_file_from_google_drive(file_id='1nfGNi7vMch6G1puGAGIokSADsc6mvyoX', dest_path='../prepped_models.tgz', overwrite=True)
	print('\nUNTARING prepped_models.tgz\n')
	call('tar -xzvf ../prepped_models.tgz',shell=True)
except:
	print('hmmm something went wrong downloading the google drive files. You can try downloading them manually through the following urls:\n')
	print('https://drive.google.com/file/d/1QlVe2_uJlHNVwYBiUr1napQtlhkJdG2_/view?usp=sharing')
	print('https://drive.google.com/file/d/1gQrMwjlIcIHleWFlt3Sb6mjcvD42wALY/view?usp=sharing')
	print('\nFrom each link you should be able to download "image_data.tgz" and "prepped_models.tgz" respectively. Untar each (tar -xvzf image_data.tgz),' +
		  'and then place the resulting folders ("image_data" and "prepped_models") in the top folder of this repo.')