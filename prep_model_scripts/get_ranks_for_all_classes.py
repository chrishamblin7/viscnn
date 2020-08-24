#get rank with respect to each class in a dataset, do this in a hacky single class way, because for some stupid reason your gpu memory is getting used up otherwise

from subprocess import call, Popen
import os
import time
import torch
import sys
sys.path.insert(0, os.path.abspath('../'))

os.chdir('../')
from prep_model_parameters import output_folder, rank_img_path
os.chdir('./prep_model_scripts')



# The node and edge ranks are fetched as a different subprocess for each label, this is done by creating a dummy dataset folder with an empty subfolder per label
#for the torchvision.datasets.ImageFolder data loader. For each label subprocess only the subfolder of that label is symlinked with data from params.rank_img_path.
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

dummy_path, labels = make_dummy_dir(rank_img_path)


#Run a subprocess for each label
print('getting node and edge ranks for all subfolders of %s'%rank_img_path)
#for label in labels:          ####EEEEEDDDDDDDITTTTTTT
for label in labels:
	print(label)
	if not os.path.exists('../prepped_models/'+output_folder+'/ranks/%s_rank.pt'%label):  #####EDDDDDIIIIITTT
		call('python get_ranks_for_single_class.py --label %s --dummy-path %s'%(label,dummy_path),shell=True)
	else:
		print('skipping %s, rank already exists'%label)


#Get rid of the dummy folder after all subprocesses have run
call('rm -r %s'%dummy_path,shell=True)




#The subprocesses have generated different files containing ranks for each label, lets now collapse them into a single dataframe
print('saving ranks to a dataframe')
start = time.time()
import pandas as pd

allnode_dflist = []
alledge_dflist = []
for label in labels:          
	ranks = torch.load('../prepped_models/'+output_folder+'/ranks/%s_rank.pt'%label)
	#nodes
	node_num = 0
	for layer in range(len(ranks['nodes']['act'])):
		for num_by_layer in range(len(ranks['nodes']['act'][layer])):
			allnode_dflist.append([node_num,layer,num_by_layer,ranks['nodes']['act'][layer][num_by_layer],ranks['nodes']['grad'][layer][num_by_layer],ranks['nodes']['weight'][layer][num_by_layer],ranks['nodes']['actxgrad'][layer][num_by_layer],label])
			node_num += 1
	#edges
	edge_num = 0
	for layer in range(len(ranks['edges']['act'])):
		for out_channel in range(len(ranks['edges']['act'][layer])):
			for in_channel in range(len(ranks['edges']['act'][layer][out_channel])):
				alledge_dflist.append([edge_num,layer,out_channel,in_channel,ranks['edges']['act'][layer][out_channel][in_channel],ranks['edges']['grad'][layer][out_channel][in_channel],ranks['edges']['weight'][layer][out_channel][in_channel],ranks['edges']['actxgrad'][layer][out_channel][in_channel],label])
				edge_num += 1


#make nodes DF
node_column_names = ['node_num','layer','node_num_by_layer','act_rank','grad_rank','weight_rank','actxgrad_rank','class']
node_df = pd.DataFrame(allnode_dflist,columns=node_column_names)
#add overall (average) rank
node_overall_list = []
for i in range(node_num):
	act_rank = sum(node_df.loc[(node_df['node_num'] == i)]['act_rank'])/len(node_df.loc[(node_df['node_num'] == i)]['act_rank'])
	grad_rank = sum(node_df.loc[(node_df['node_num'] == i)]['grad_rank'])/len(node_df.loc[(node_df['node_num'] == i)]['grad_rank'])
	weight_rank = sum(node_df.loc[(node_df['node_num'] == i)]['weight_rank'])/len(node_df.loc[(node_df['node_num'] == i)]['weight_rank'])
	actxgrad_rank = sum(node_df.loc[(node_df['node_num'] == i)]['actxgrad_rank'])/len(node_df.loc[(node_df['node_num'] == i)]['actxgrad_rank'])
	layer = node_df.loc[(node_df['node_num'] == i)]['layer'].iloc[0]
	node_num_by_layer = node_df.loc[(node_df['node_num'] == i)]['node_num_by_layer'].iloc[0]
	node_overall_list.append([i,layer,node_num_by_layer,act_rank,grad_rank,weight_rank,actxgrad_rank,'overall'])

node_overall_df = pd.DataFrame(node_overall_list,columns=node_column_names)
node_df = node_df.append(node_overall_df)

#save
node_df.to_csv('../prepped_models/'+output_folder+'/node_ranks.csv',index=False)

#make edges DF
edge_column_names = ['edge_num','layer','out_channel','in_channel','act_rank','grad_rank','weight_rank','actxgrad_rank','class']
edge_df = pd.DataFrame(alledge_dflist,columns=edge_column_names)
#add overall (average) rank
edge_overall_list = []
for i in range(edge_num):
	act_rank = sum(edge_df.loc[(edge_df['edge_num'] == i)]['act_rank'])/len(edge_df.loc[(edge_df['edge_num'] == i)]['act_rank'])
	grad_rank = sum(edge_df.loc[(edge_df['edge_num'] == i)]['grad_rank'])/len(edge_df.loc[(edge_df['edge_num'] == i)]['grad_rank'])
	weight_rank = sum(edge_df.loc[(edge_df['edge_num'] == i)]['weight_rank'])/len(edge_df.loc[(edge_df['edge_num'] == i)]['weight_rank'])
	actxgrad_rank = sum(edge_df.loc[(edge_df['edge_num'] == i)]['actxgrad_rank'])/len(edge_df.loc[(edge_df['edge_num'] == i)]['actxgrad_rank'])
	layer = edge_df.loc[(edge_df['edge_num'] == i)]['layer'].iloc[0]
	out_channel = edge_df.loc[(edge_df['edge_num'] == i)]['out_channel'].iloc[0]
	in_channel = edge_df.loc[(edge_df['edge_num'] == i)]['in_channel'].iloc[0]
	edge_overall_list.append([i,layer,out_channel,in_channel,act_rank,grad_rank,weight_rank,actxgrad_rank,'overall'])

edge_overall_df = pd.DataFrame(edge_overall_list,columns=edge_column_names)
edge_df = edge_df.append(edge_overall_df)

#save
edge_df.to_csv('../prepped_models/'+output_folder+'/edge_ranks.csv',index=False)

print('save dataframe time: %s'%(time.time()-start))

# #remove ranks folder with individual label ranks
# call('rm -r ../prepped_models/%s/ranks/'%(output_folder),shell=True)
