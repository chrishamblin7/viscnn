import os
from subprocess import call
from prep_model_parameters import output_folder, save_activations
import time

#Set up output directory
go = True

print('setting up output directory prepped_models/'+output_folder)
if not os.path.exists('prepped_models/'+output_folder):
	os.mkdir('prepped_models/'+output_folder)
	#deeviz folders
	os.mkdir('prepped_models/'+output_folder+'/visualizations/')
	os.mkdir('prepped_models/'+output_folder+'/visualizations/images')
	with open('prepped_models/'+output_folder+'/visualizations/images.csv', 'a') as images_csv:
		images_csv.write('image_name,nodeid,obj,parametrizer,optimizer,tranforms\n')

else:
	print('prepped_models/%s already exists! It will be overwritten!'%output_folder)
	resp = input('Are you sure you want to continue and overwrite this folder [y/n]:')
	if resp not in ['y','Y','yes','Yes','YES','TRUE','True','true']:
		go = False

if not go:
	exit()

start= time.time()

call(['cp','prep_model_parameters.py','prepped_models/'+output_folder+'/prep_model_params_used.py'])

#switch into scripts directory
print('changing directory to prep_model_scripts')
os.chdir('prep_model_scripts/')

#kernels
print('pulling model convolutional kernels . . .')
call(['python', 'get_kernels.py'])

#activation maps
if save_activations:
	print('getting node and edge activation maps for input images')
	call(['python','get_activation_maps_for_input_images.py'])
else:
	print('save_activations = False in parameter file, not fetching activations for input images.')
	
#ranks
print('getting node and edge subgraph importance ranks')
call(['python','get_ranks_for_all_categories.py'])
call(['python','gen_overall_rank.py'])
call(['python','gen_categories_rank_df.py'])

#misc graph data
print('getting miscellaneous graph data')
call(['python','get_misc_graph_data.py'])

#graph node and edge positions
print('generating positions of nodes and edges in graph')
call(['python','get_graph_positions.py'])

print('Run Time: %s'%str(time.time()-start))

