import os
from subprocess import call


print('Hello! Set up consists of two parts; setting up a python environment,' +
	  'and downloading files that are two big to store on github. These parts can be run ' +
	  'separately based on your input.')


#Environment Setup
try:
	resp = input('Would you like to setup a new python environment for this project (y/n): ')
except:
	resp = raw_input('Would you like to setup a new python environment for this project (y/n): ')
if resp in ['y','Y','yes','Yes','YES','TRUE','True','true']:
	try:
		print('Setting up python virtual environment')
		call('python3 -m venv env && source env/bin/activate && pip install -r setup_scripts/subgraph_visualizer_pip_requirements.txt',shell=True)
		print('virtualenv created! Activate it with "source env/bin/activate" before running scripts in this repository.')
	except:
		print('hmmmm something went wrong installing the environment. Maybe try setting up the environment manual using "setup_scripts/subgraph_visualizer_pip_requirements.txt" ' +
			  'as a reference. Try "pip install -r setup_scripts/subgraph_visualizer_pip_requirements.txt" Most of the packages there-in are dependencies, the only packages you should have to directly install are:')
		print('pip install torchvision')
		print('pip install pandas')
		print('pip install -U scikit-learn')
		print('pip install plotly')
		print('pip install dash')
		print('pip install googledrivedownloader')



#Large Folder Downloading
print('\nThis repository contains files for models already prepped for visualization ' +
	  'that must be downloaded from google drive.')
try:
	resp = input('Would you like to download these files from google drive now (y/n): ')
except:
	resp = raw_input('Would you like to download these files from google drive now (y/n): ')
if resp in ['y','Y','yes','Yes','YES','TRUE','True','true']:
	call('source env/bin/activate && python setup_scripts/download_from_gdrive.py',shell=True)
else:
	if not os.path.exists('./image_data'):
		os.mkdir('./image_data')
	if not os.path.exists('./prepped_models'):
		os.mkdir('./prepped_models')

print('all done!')





