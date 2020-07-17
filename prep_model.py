import os
from subprocess import call
from prep_model_parameters import output_folder


#Set up output directory
go = True

print('setting up output directory prepped_models/'+output_folder)
if not os.path.exists('prepped_models/'+output_folder):
	os.mkdir('prepped_models/'+output_folder)
else:
	print('prepped_models/%s already exists! It will be overwritten!'%output_folder)
	resp = input('Are you sure you want to continue and overwrite this folder [y/n]:')
	if resp not in ['y','Y','yes','Yes','YES','TRUE','True','true']:
		go = False

if not go:
	exit()

call(['cp','prep_model_parameters.py','prepped_models/'+output_folder+'/prep_model_params_used.py'])

#switch into scripts directory
print('changing directory to prep_model_scripts')
os.chdir('prep_model_scripts/')

#kernels
print('pulling model convolutional kernels . . .')
call(['python', 'get_kernels.py'])

#activation maps
print('getting node and edge activation maps for input images')
call(['python','get_activation_maps_for_input_images.py'])

#ranks
print('getting node and edge subgraph importance ranks')
call(['python','get_ranks_for_all_classes.py'])




