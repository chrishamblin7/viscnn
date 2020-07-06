import os
from suprocess import call


print('Hello! Set up consists of two parts; setting up a python environment,' +
	  'and downloading files that are two big to store on github. These parts can be run' +
	  'separately based on your input.')


#Environment Setup
resp = raw_input('Would you like to setup a new python environment for this project (y/n): ')
if resp in ['y','Y','yes','Yes','YES','TRUE','True','true']:
	try:
		print('Setting up python environment')
		resp = raw_input('Do you have a conda package manager and would like to use it (y/n): ')
		if resp in ['y','Y','yes','Yes','YES','TRUE','True','true']:
			print('Okay, setting up an environment with conda.')
			call('conda env create -f setup_scripts/subgraph_visualizer_environment.yml',shell=True)
			print('conda environment create! To run the scripts in this repo, first run "conda activate subgraph_visualizer"')

		else:
			print('Okay, setting up an environment with virtualenv instead of conda.')
			call('python3 -m venv env',shell=True)
			call('source env/bin/activate',shell=True)
			call('pip install -r setup_scripts/subgraph_visualizer_pip_requirements.txt',shell=True)
			print('virtualenv created! Activate it with "source env/bin/activate" before running scripts in this repository.')
	except:
		print('hmmmm something went wrong installing the environment. Maybe try setting up the environment manual using "setup_scripts/subgraph_visualizer_pip_requirements.txt"' +
			  'as a reference. Try "pip install -r setup_scripts/subgraph_visualizer_pip_requirements.txt" Most of the packages there-in are dependencies, the only packages you should have to directly install are:')
		print('pip install torchvision')
		print('pip install pandas')
		print('pip install -U scikit-learn')
		print('pip install plotly')
		print('pip install dash')



#Large Folder Downloading
print('\nThis repository contains files for models already prepped for visualization' +
	  'that must be downloaded from google drive.')
resp = raw_input('Would you like to download these files from google drive now (y/n): ')
if resp in ['y','Y','yes','Yes','YES','TRUE','True','true']:
	try:
		print('LARGE FOLDER DOWNLOAD\n')
		print('DOWNLOADING IMAGE-DATA FROM GDRIVE\n')
		call('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1QlVe2_uJlHNVwYBiUr1napQtlhkJdG2_\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p\')&id=1QlVe2_uJlHNVwYBiUr1napQtlhkJdG2_" -O image_data.tgz && rm -rf /tmp/cookies.txt',shell=True)
		print('\nUNTARING image_data.tgz\n')
		call('tar -xvzf image_data.tgz',shell=True)

		print('DOWNLOADING PREPPED-MODELS FROM GDRIVE\n')
		call('wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id=1gQrMwjlIcIHleWFlt3Sb6mjcvD42wALY\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p\')&id=1gQrMwjlIcIHleWFlt3Sb6mjcvD42wALY" -O prepped_models.tgz && rm -rf /tmp/cookies.txt',shell=True)
		print('\nUNTARING prepped_models.tgz\n')
		call('tar -xvzf prepped_models.tgz',shell=True)
	except:
		print('hmmm something went wrong downloading the google drive files. You can try downloading them manually through the following urls:\n')
		print('https://drive.google.com/file/d/1QlVe2_uJlHNVwYBiUr1napQtlhkJdG2_/view?usp=sharing')
		print('https://drive.google.com/file/d/1gQrMwjlIcIHleWFlt3Sb6mjcvD42wALY/view?usp=sharing')
		print('\nFrom each link you should be able to download "image_data.tgz" and "prepped_models.tgz" respectively. Untar each (tar -xvzf image_data.tgz),' +
			  'and then place the resulting folders ("image_data" and "prepped_models") in the top folder of this repo.')
else:
	if not os.path.exists('./image_data'):
		os.mkdir('./image_data')
	if not os.path.exists('./prepped_models'):
		os.mkdir('./prepped_models')

print('all done!')








