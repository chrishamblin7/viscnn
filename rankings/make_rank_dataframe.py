#make a pandas dataframe object of rankings
import os
import torch
import pandas as pd


'''
file_dict = {'animate':torch.load('model_animate_imgnet_6_epochs_rank_v2.pt'),
			 'inanimate':torch.load('model_inanimate_imgnet_6_epochs_rank_v2.pt'),
			 'letters':torch.load('model_letters_imgnet_6_epochs_rank_v2.pt'),
			 'faces':torch.load('model_faces_50adam_lr.001_rank.pt'),
			  'enumeration':torch.load('model_enumeration_rank.pt')}
'''
labels = os.listdir('/home/chris/projects/categorical_pruning/data/cifar10/train')
labels.append('overall')

biglist = []

for label in labels:
	ranks = torch.load('cifar/%s_rank.pt'%(label))
	for i in range(len(ranks)):
		biglist.append([ranks[i][0],ranks[i][1],ranks[i][2],ranks[i][3],label])

column_names = ['filter_num','layer','filter_num_by_layer','prune_score','class']

df = pd.DataFrame(biglist,columns=column_names)
df.to_csv('cifar_prunned_ranks.csv',index=False)
#df.to_feather('letnum_small_nopool_ranks.feather')
