#get rank with respect to each category in a dataset, do this in a hacky single category way, because for some stupid reason your gpu memory is getting used up otherwise

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


if not os.path.exists('../prepped_models/'+output_folder):
	os.mkdir('../prepped_models/'+output_folder)

categories = []
for subdir in os.listdir(rank_img_path):
	if os.path.isdir(os.path.join(rank_img_path,subdir)):
		categories.append(subdir)

#Run a subprocess for each category
print('getting node and edge ranks for all subfolders of %s'%rank_img_path)

for category in categories:
	print(category)
	if not os.path.exists('../prepped_models/'+output_folder+'/ranks/%s_rank.pt'%category):  #####EDDDDDIIIIITTT
		call('python get_ranks_for_single_category.py --category %s --data-path %s'%(category,os.path.join(rank_img_path,category)),shell=True)
	else:
		print('skipping %s, rank already exists'%category)


#Get rid of the dummy folder after all subprocesses have run
#call('rm -r %s'%dummy_path,shell=True)


