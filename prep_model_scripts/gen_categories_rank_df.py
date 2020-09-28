#combine all categories ranks into single dataframe

from subprocess import call, Popen
import os
import time
import torch
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath('../'))

os.chdir('../')
from prep_model_parameters import output_folder, rank_img_path, save_activations
os.chdir('./prep_model_scripts')



#Make dataframes of rank files
print('saving category ranks to a dataframe')
print('node dataframe')
start = time.time()



categories = []
for subdir in os.listdir(rank_img_path):
	if os.path.isdir(os.path.join(rank_img_path,subdir)):
		categories.append(subdir)
categories.sort()
categories.insert(0,'overall')

#Nodes
allnode_dflist = []

for category in categories:         
	ranks = torch.load('../prepped_models/'+output_folder+'/ranks/categories_nodes/%s_nodes_rank.pt'%category)
	#nodes
	node_num = 0
	for layer in range(len(ranks['act']['prenorm'])):
		for num_by_layer in range(len(ranks['act']['prenorm'][layer])):
			allnode_dflist.append([node_num,layer,num_by_layer,ranks['act']['prenorm'][layer][num_by_layer],ranks['grad']['prenorm'][layer][num_by_layer],ranks['actxgrad']['prenorm'][layer][num_by_layer],
								   ranks['act']['norm'][layer][num_by_layer],ranks['grad']['norm'][layer][num_by_layer],ranks['actxgrad']['norm'][layer][num_by_layer],category])
			node_num += 1
#make nodes DF
node_column_names = ['node_num','layer','node_num_by_layer','act_prenorm_rank','grad_prenorm_rank','actxgrad_prenorm_rank',
					 'act_norm_rank','grad_norm_rank','actxgrad_norm_rank','category']
node_df = pd.DataFrame(allnode_dflist,columns=node_column_names)
#save
node_df.to_csv('../prepped_models/'+output_folder+'/ranks/categories_nodes_ranks.csv',index=False)


#make edges DF
if save_activations:     #dont save edge dataframe if using dynamic input, too much memory
	alledge_dflist = []
	for category in categories:
		ranks = torch.load('../prepped_models/'+output_folder+'/ranks/categories_edges/%s_edges_rank.pt'%category)		
		edge_num = 0
		for layer in range(len(ranks['act']['prenorm'])):
			for out_channel in range(len(ranks['act']['prenorm'][layer])):
				for in_channel in range(len(ranks['act']['prenorm'][layer][out_channel])):
					alledge_dflist.append([edge_num,layer,out_channel,in_channel,ranks['act']['prenorm'][layer][out_channel][in_channel],ranks['grad']['prenorm'][layer][out_channel][in_channel],ranks['actxgrad']['prenorm'][layer][out_channel][in_channel],
										   ranks['act']['norm'][layer][out_channel][in_channel],ranks['grad']['norm'][layer][out_channel][in_channel],ranks['actxgrad']['norm'][layer][out_channel][in_channel],category])
					edge_num += 1

	edge_column_names = ['edge_num','layer','out_channel','in_channel','act_prenorm_rank','grad_prenorm_rank','actxgrad_prenorm_rank',
					 	 'act_norm_rank','grad_norm_rank','actxgrad_norm_rank','category']
	edge_df = pd.DataFrame(alledge_dflist,columns=edge_column_names)
	#save
	edge_df.to_csv('../prepped_models/'+output_folder+'/ranks/categories_edges_ranks.csv',index=False)

print('save dataframe time: %s'%(time.time()-start))

# #remove ranks folder with individual category ranks
# call('rm -r ../prepped_models/%s/ranks/'%(output_folder),shell=True)