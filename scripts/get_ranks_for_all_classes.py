#get rank with respect to each class in a dataset, do this in a hacky single class way, because for some stupid reason your gpu memory is getting used up otherwise


from subprocess import call, Popen
import os
import argparse
import time
import parameters as params


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

print('getting node and edge ranks for all subfolders of %s'%params.rank_img_path)
for label in labels:
	print(label)

	#call('CUDA_VISIBLE_DEVICES=%s python single_class.py --label %s --dummy_path %s --train_path %s --model %s --output_folder %s %s %s'%(args.device,label,dummy_path,args.train_path,args.model,args.output_folder,grayscale,no_crop),shell=True)
	call(' python get_ranks_for_single_class.py --label %s --dummy-path %s'%(label,dummy_path),shell=True)

call('rm -r %s'%dummy_path,shell=True)

