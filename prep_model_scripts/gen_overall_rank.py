import time
import torch
import os

overall = None

ranks_folder = '/mnt/data/chris/dropbox/Research-Hamblin/Projects/cnn_subgraph_visualizer/prepped_models/alexnet_200/ranks/'

rank_files = os.listdir(ranks_folder)

rank_files.sort()

for rank_file in rank_files:
    print(rank_file)
    rank_dict = torch.load(os.path.join(ranks_folder,rank_file))
    if overall is None:  #init
        overall = rank_dict
    else:   #sum all ranks together pointwise
        for part in ['nodes','edges']:
            for rank_type in ['actxgrad','act','grad','weight']:
                for i in range(len(overall[part][rank_type])):
                    overall[part][rank_type][i] = overall[part][rank_type][i] + rank_dict[part][rank_type][i]

#average by dividing by number of ranks
for part in ['nodes','edges']:
    for rank_type in ['actxgrad','act','grad','weight']:
        for i in range(len(overall[part][rank_type])):
            overall[part][rank_type][i] = overall[part][rank_type][i]/len(rank_files)

torch.save(overall,os.path.join(ranks_folder,'overall_rank.pt'))