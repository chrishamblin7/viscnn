#get rank with respect to each class in a dataset, do this in a hacky single class way, because for some stupid reason your gpu memory is getting used up otherwise


from subprocess import call, Popen
import os
import time
import parameters as params



# The node and edge ranks are fetched as a different subprocess for each label, this is done by creating a dummy dataset folder with an empty subfolder per label
#for the torchvision.datasets.ImageFolder data loader. For each label subprocess only the subfolder of that label is symlinked with data from params.rank_img_path.
def make_dummy_dir(orig_path):
	path_split = orig_path.split('/')
	path_split[-1] = path_split[-1] + '_dummy'
	dummy_path = '/'.join(path_split)
	os.makedirs(dummy_path,exist_ok=True)
	subdirs = os.listdir(orig_path)
	for subdir in subdirs:
		os.makedirs(os.path.join(dummy_path,subdir),exist_ok=True)
	return dummy_path, subdirs


if not os.path.exists('../prepped_models/'+params.output_folder):
	os.mkdir('../prepped_models/'+params.output_folder)

dummy_path, labels = make_dummy_dir(params.rank_img_path)


#Run a subprocess for each label
print('getting node and edge ranks for all subfolders of %s'%params.rank_img_path)
for label in labels:
	print(label)

	#call('CUDA_VISIBLE_DEVICES=%s python single_class.py --label %s --dummy_path %s --train_path %s --model %s --output_folder %s %s %s'%(args.device,label,dummy_path,args.train_path,args.model,args.output_folder,grayscale,no_crop),shell=True)
	call(' python get_ranks_for_single_class.py --label %s --dummy-path %s'%(label,dummy_path),shell=True)


#Get rid of the dummy folder after all subprocesses have run
call('rm -r %s'%dummy_path,shell=True)




#The subprocesses have generated different files containing ranks for each label, lets now collapse them into a single dataframe
print('saving ranks to a dataframe')
import pandas as pd

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
#add overall (average) rank
node_overall_list = []
for i in range(len(node_num)):
	rank = sum(node_df.loc[(node_df['node_num'] == i)]['rank_score'])/len(node_df.loc[(node_df['node_num'] == i)]['rank_score'])
	layer = node_df.loc[(node_df['node_num'] == i)]['layer'].iloc[0]
	node_num_by_layer = node_df.loc[(node_df['node_num'] == i)]['node_num_by_layer'].iloc[0]
	node_overall_list.append([i,layer,node_num_by_layer,rank,'overall'])

node_overall_df = pd.DataFrame(node_overall_list,columns=node_column_names)
node_df = node_df.append(node_overall_df)

#save
node_df.to_csv('../prepped_models/'+params.output_folder+'/node_ranks.csv',index=False)

#make edges DF
edge_column_names = ['edge_num','layer','out_channel','in_channel','rank_score','class']
edge_df = pd.DataFrame(alledge_dflist,columns=edge_column_names)
#add overall (average) rank
edge_overall_list = []
for i in range(len(edge_num)):
	rank = sum(edge_df.loc[(edge_df['edge_num'] == i)]['rank_score'])/len(edge_df.loc[(edge_df['edge_num'] == i)]['rank_score'])
	layer = edge_df.loc[(edge_df['edge_num'] == i)]['layer'].iloc[0]
	out_channel = edge_df.loc[(edge_df['edge_num'] == i)]['out_channel'].iloc[0]
	in_channel = edge_df.loc[(edge_df['edge_num'] == i)]['in_channel'].iloc[0]
	edge_overall_list.append([i,layer,out_channel,in_channel,rank,'overall'])

edge_overall_df = pd.DataFrame(edge_overall_list,columns=edge_column_names)
edge_df = edge_df.append(edge_overall_df)

#save
edge_df.to_csv('../prepped_models/'+params.output_folder+'/edge_ranks.csv',index=False)


#remove ranks folder with individual label ranks
#call('rm -r ../prepped_models/%s/ranks/'%(params.output_folder),shell=True)
