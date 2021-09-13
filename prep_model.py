import os
from subprocess import call
from prep_model_parameters import output_folder, save_activations, save_node_visualizations, save_edge_visualizations, deepviz_projections
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
		images_csv.write('image_name,targetid,objective,parametrizer,optimizer,transforms,neuron\n')

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
call(['python', 'get_kernels.py',output_folder])

#activation maps
if save_activations:
	print('getting node and edge activation maps for input images')
	call(['python','get_activation_maps_for_input_images.py',output_folder])
else:
	print('save_activations = False in parameter file, not fetching activations for input images.')
	
#ranks
print('getting node and edge subgraph importance ranks')
call(['python','get_ranks_for_all_categories.py',output_folder])
call(['python','gen_overall_rank.py',output_folder])
call(['python','gen_categories_rank_df.py',output_folder])

#misc graph data
print('getting miscellaneous graph data')
call(['python','get_misc_graph_data.py',output_folder])


#deep visualizations
if save_node_visualizations:
	call(['python','gen_deepviz_for_nodes.py', output_folder])

if save_edge_visualizations:
	call(['python','gen_deepviz_for_edges.py', output_folder])


#generate graph projections using deep visualizations as basis
if deepviz_projections:
	call(['python','gen_model_activations_from_deepviz.py', output_folder])

#graph node and edge positions
print('generating positions of nodes and edges in graph')
call(['python','get_graph_positions.py', output_folder])

print('Run Time: %s'%str(time.time()-start))

