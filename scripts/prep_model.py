import os
from subprocess import call
from params import output_folder


#Set up output directory
print('setting up output directory ../prepped_models/'+output_folder)
if not os.path.exists('../prepped_models/'+output_folder):
	os.mkdir('../prepped_models/'+output_folder)
call(['cp','parameters.py','../prepped_models/'+output_folder+'/prep_model_params_used.py'])

#kernels
print('pulling model convolutional kernels . . .')
call(['python', 'get_kernels.py'])

#activation maps
print('getting node and edge activation maps for input images')
call(['python','get_activation_maps_for_input_images.py'])

#ranks
print('getting node and edge subgraph importance ranks')
call(['python','get_ranks_for_all_classes.py'])




