print('saving ranks to a dataframe')
import pandas as pd
import parameters as params
import os
import torch

labels = os.listdir('/home/chris/projects/categorical_pruning/pruning/data/cifar10/train')


allnode_dflist = []
alledge_dflist = []
for label in labels:
	ranks = torch.load('../prepped_models/'+params.output_folder+'/ranks/%s_rank.pt'%label)
	#nodes
	node_num = 0
	for layer in range(len(ranks['nodes'])):
		for num_by_layer in range(len(ranks['nodes'][layer])):
			allnode_dflist.append([node_num,layer,num_by_layer,ranks['nodes'][layer][num_by_layer],label])
			node_num += 1
	#edges
	edge_num = 0
	for layer in range(len(ranks['edges'])):
		for out_channel in range(len(ranks['edges'][layer])):
			for in_channel in range(len(ranks['edges'][layer][out_channel])):
				alledge_dflist.append([edge_num,layer,out_channel,in_channel,ranks['edges'][layer][out_channel][in_channel],label])
			edge_num += 1




#make nodes DF
node_column_names = ['node_num','layer','node_num_by_layer','rank_score','class']
node_df = pd.DataFrame(allnode_dflist,columns=node_column_names)
print(node_df.head(10))
#add overall (average) rank
node_overall_list = []
for i in range(node_num):
	print(i)
	rank = sum(node_df.loc[(node_df['node_num'] == i)]['rank_score'])/len(node_df.loc[(node_df['node_num'] == i)]['rank_score'])
	layer = node_df.loc[(node_df['node_num'] == i)]['layer'].iloc[0]
	node_num_by_layer = node_df.loc[(node_df['node_num'] == i)]['node_num_by_layer'].iloc[0]
	node_overall_list.append([i,layer,node_num_by_layer,rank,'overall'])

node_overall_df = pd.DataFrame(node_overall_list,columns=node_column_names)
node_df = node_df.append(node_overall_df)

#save
node_df.to_csv('../prepped_models/'+params.output_folder+'node_ranks.csv',index=False)

#make edges DF
edge_column_names = ['edge_num','layer','out_channel','in_channel','rank_score','class']
edge_df = pd.DataFrame(alledge_dflist,columns=edge_column_names)
print(edge_df.head(10))
#add overall (average) rank
edge_overall_list = []
for i in range(edge_num):
	print(i)
	rank = sum(edge_df.loc[(edge_df['edge_num'] == i)]['rank_score'])/len(edge_df.loc[(edge_df['edge_num'] == i)]['rank_score'])
	layer = edge_df.loc[(edge_df['edge_num'] == i)]['layer'].iloc[0]
	out_channel = edge_df.loc[(edge_df['edge_num'] == i)]['out_channel'].iloc[0]
	in_channel = edge_df.loc[(edge_df['edge_num'] == i)]['in_channel'].iloc[0]
	edge_overall_list.append([i,layer,out_channel,in_channel,rank,'overall'])

edge_overall_df = pd.DataFrame(edge_overall_list,columns=edge_column_names)
edge_df = edge_df.append(edge_overall_df)

#save
edge_df.to_csv('../prepped_models/'+params.output_folder+'edge_ranks.csv',index=False)


#remove ranks folder with individual label ranks
#call('rm -r ../prepped_models/%s/ranks/'%(params.output_folder),shell=True)