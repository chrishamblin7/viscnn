
#file generates an category 'overall' rank file, which is the average of all other categories
from subprocess import call, Popen
import os
import time
import torch
import pandas as pd
import argparse
import sys

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("output_folder", type = str, help='the folder name for this prepped model')
	args = parser.parse_args()
	return args

args = get_args()
output_folder = args.output_folder

sys.path.insert(0, os.path.abspath('../prepped_models/%s'%output_folder))

os.chdir(os.path.abspath('../prepped_models/%s'%output_folder))
from prep_model_params_used import rank_img_path, save_activations
os.chdir('../../prep_model_scripts')


#overall rank
print('generating rank of "overall" category, averaging all other ranks')

ranks_folder = '../prepped_models/%s/ranks/'%output_folder

if os.path.exists(os.path.join(ranks_folder,'categories_nodes','overall_nodes_rank.pt')):
	print('overall rank file already exists')
	exit()


for part in ['nodes','edges']:
	overall = None
	rank_files = os.listdir(os.path.join(ranks_folder,'categories_%s'%part))
	rank_files.sort()
	for rank_file in rank_files:
		rank_dict = torch.load(os.path.join(ranks_folder,'categories_%s'%part,rank_file))
		if overall is None:  #init
			overall = rank_dict
		else:   #sum all ranks together pointwise
			for rank_type in ['actxgrad','act','grad']:
				for i in range(len(overall[rank_type])):
					overall[rank_type][i][1] = overall[rank_type][i][1] + rank_dict[rank_type][i][1]
	#average by dividing by number of ranks
	for rank_type in ['actxgrad','act','grad']:
		for i in range(len(overall[rank_type])):
			overall[rank_type][i][1] = overall[rank_type][i][1]/len(rank_files)
	#save file
	torch.save(overall,os.path.join(ranks_folder,'categories_%s'%part,'overall_%s_rank.pt'%part))