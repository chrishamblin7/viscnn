#get rank with respect to each category in a dataset, do this in a hacky single category way, because for some stupid reason your gpu memory is getting used up otherwise

from subprocess import call, Popen
import os
import time
import torch
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath('../'))

os.chdir('../')
from prep_model_parameters import output_folder, rank_img_path, dynamic_input
os.chdir('./prep_model_scripts')



# The node and edge ranks are fetched as a different subprocess for each category, this is done by creating a dummy dataset folder with an empty subfolder per category
#for the torchvision.datasets.ImageFolder data loader. For each category subprocess only the subfolder of that category is symlinked with data from params.rank_img_path.
def make_dummy_dir(orig_path):
	path_split = orig_path.split('/')
	path_split[-1] = path_split[-1] + '_dummy'
	dummy_path = '/'.join(path_split)
	if os.path.exists(dummy_path):
		call('rm -r %s'%dummy_path,shell=True)
	os.makedirs(dummy_path,exist_ok=True)
	subdirs = []
	for fname in os.listdir(orig_path):
		path = os.path.join(orig_path, fname)
		if os.path.isdir(path):
			subdirs.append(fname)
	subdirs.sort()
	for subdir in subdirs:
		os.makedirs(os.path.join(dummy_path,subdir),exist_ok=True)
	return dummy_path, subdirs


if not os.path.exists('../prepped_models/'+output_folder):
	os.mkdir('../prepped_models/'+output_folder)

dummy_path, categories = make_dummy_dir(rank_img_path)


#Run a subprocess for each category
print('getting node and edge ranks for all subfolders of %s'%rank_img_path)

for category in categories:
	print(category)
	if not os.path.exists('../prepped_models/'+output_folder+'/ranks/%s_rank.pt'%category):  #####EDDDDDIIIIITTT
		call('python get_ranks_for_single_category.py --category %s --dummy-path %s'%(category,dummy_path),shell=True)
	else:
		print('skipping %s, rank already exists'%category)


#Get rid of the dummy folder after all subprocesses have run
call('rm -r %s'%dummy_path,shell=True)


#overall rank
print('generating rank of "overall" category, averaging all other ranks')

ranks_folder = '../prepped_models/%s/ranks/'%output_folder
for part in ['nodes','edges']:
	overall = None
	rank_files = os.listdir(os.path.join(ranks_folder,part))
	rank_files.sort()
	for rank_file in rank_files:
		rank_dict = torch.load(os.path.join(ranks_folder,part,rank_file))
		if overall is None:  #init
			overall = rank_dict
		else:   #sum all ranks together pointwise
			for rank_type in ['actxgrad','act','grad','weight']:
				for i in range(len(overall[rank_type])):
					overall[rank_type][i] = overall[rank_type][i] + rank_dict[rank_type][i]
	#average by dividing by number of ranks
	for rank_type in ['actxgrad','act','grad','weight']:
		for i in range(len(overall[rank_type])):
			overall[rank_type][i] = overall[rank_type][i]/len(rank_files)
	#save file
	torch.save(overall,os.path.join(ranks_folder,part,'overall_%s_rank.pt'%part))


#Make dataframes of rank files
print('saving ranks to a dataframe')
print('node dataframe')
start = time.time()

categories.insert(0,'overall')

allnode_dflist = []

for category in categories:         
	ranks = torch.load('../prepped_models/'+output_folder+'/ranks/nodes/%s_nodes_rank.pt'%category)
	#nodes
	node_num = 0
	for layer in range(len(ranks['act'])):
		for num_by_layer in range(len(ranks['act'][layer])):
			allnode_dflist.append([node_num,layer,num_by_layer,ranks['act'][layer][num_by_layer],ranks['grad'][layer][num_by_layer],ranks['weight'][layer][num_by_layer],ranks['actxgrad'][layer][num_by_layer],category])
			node_num += 1

#make nodes DF
node_column_names = ['node_num','layer','node_num_by_layer','act_rank','grad_rank','weight_rank','actxgrad_rank','category']
node_df = pd.DataFrame(allnode_dflist,columns=node_column_names)
#save
node_df.to_csv('../prepped_models/'+output_folder+'/node_ranks.csv',index=False)

#make edges DF
if not dynamic_input:     #dont save edge dataframe if using dynamic input, too much memory
	alledge_dflist = []
	for category in categories:
		ranks = torch.load('../prepped_models/'+output_folder+'/ranks/edges/%s_edges_rank.pt'%category)		
		edge_num = 0
		for layer in range(len(ranks['act'])):
			for out_channel in range(len(ranks['act'][layer])):
				for in_channel in range(len(ranks['act'][layer][out_channel])):
					alledge_dflist.append([edge_num,layer,out_channel,in_channel,ranks['act'][layer][out_channel][in_channel],ranks['grad'][layer][out_channel][in_channel],ranks['weight'][layer][out_channel][in_channel],ranks['actxgrad'][layer][out_channel][in_channel],category])
					edge_num += 1

	edge_column_names = ['edge_num','layer','out_channel','in_channel','act_rank','grad_rank','weight_rank','actxgrad_rank','category']
	edge_df = pd.DataFrame(alledge_dflist,columns=edge_column_names)
	#save
	edge_df.to_csv('../prepped_models/'+output_folder+'/edge_ranks.csv',index=False)

print('save dataframe time: %s'%(time.time()-start))

# #remove ranks folder with individual category ranks
# call('rm -r ../prepped_models/%s/ranks/'%(output_folder),shell=True)
