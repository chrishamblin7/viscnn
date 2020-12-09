import os
import time
import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.abspath('../'))

os.chdir('../')
from prep_model_parameters import output_folder
os.chdir('./prep_model_scripts')

data = {}

nodes_df = pd.read_csv('../prepped_models/%s/ranks/categories_nodes_ranks.csv'%output_folder)


#list of layer nodes
layer_nodes_dict = {}

for row in nodes_df[nodes_df['category'] == 'overall'].itertuples(): 
    if row.layer not in layer_nodes_dict:
        layer_nodes_dict[row.layer] = [row.layer_name,[]]
    layer_nodes_dict[row.layer][1].append(row.node_num)
layer_nodes = []
for l in range(len(layer_nodes_dict)):
    layer_nodes.append(layer_nodes_dict[l])





num_layers = max(layer_nodes_dict.keys()) + 1
num_nodes = len(nodes_df.loc[nodes_df['category']=='overall'])

#list of categories
categories = list(nodes_df['category'].unique())
categories.remove('overall')
categories.insert(0,'overall')


#edges
overall_edge_ranks = torch.load('../prepped_models/%s/ranks/categories_edges/overall_edges_rank.pt'%output_folder)

num_img_chan = overall_edge_ranks['actxgrad']['prenorm'][0][1].shape[1]   #number of channels in input image

#imgnode data
layer_distance=1

def gen_imgnode_graphdata(num_chan = num_img_chan, layer_distance=1):     #returns positions, colors and names for imgnode graph points
    if num_chan == 1: #return a centered position, grey square, with 'gs' label
        return {'X':[-1*layer_distance],'Y':[0],'Z':[0]}, ['rgba(170,170,170,.7)'], ['gs']
    if num_chan == 3:
        colors = ['rgba(255,0,0,.7)','rgba(0,255,0,.7)','rgba(0,0,255,.7)']
        names = ['r','g','b']
    else:
        #colors
        other_colors = ['rgba(255,0,0,.7)','rgba(0,255,0,.7)','rgba(0,0,255,.7)',
                        'rgba(255,150,0,.7)','rgba(0,255,150,.7)','rgba(150,0,255,.7)',
                        'rgba(255,0,150,.7)','rgba(150,255,0,.7)','rgba(0,150,255,.7)']
        colors = []
        for i in num_chan:
            colors.append(i%len(other_colors)) 
        #names
        names = []
        for i in range(num_chan):
            names.append('img_'+str(i))   
            
    positions = {'X':[],'Y':[],'Z':[]}     #else return points evenly spaced around a unit circle
    a = 2*np.pi/num_chan          #angle to rotate each point
    for p in range(num_chan):
        positions['X'].append(-1*layer_distance)
        positions['Y'].append(round(np.sin(a*p)/5,2))
        positions['Z'].append(round(np.cos(a*p)/5,2)) 
    
    return positions, colors, names

imgnode_positions,imgnode_colors,imgnode_names = gen_imgnode_graphdata()

data['layer_nodes'] = layer_nodes
data['layer_nodes_dict'] = layer_nodes_dict
data['num_layers'] = num_layers
data['num_nodes'] = num_nodes
data['categories'] = categories
data['num_img_chan'] = num_img_chan
data['imgnode_positions'] = imgnode_positions
data['imgnode_colors'] = imgnode_colors
data['imgnode_names'] = imgnode_names

import pickle
pickle.dump(data,open('../prepped_models/%s/misc_graph_data.pkl'%output_folder,'wb'))