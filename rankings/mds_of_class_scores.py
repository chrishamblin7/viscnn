#layerwise mds


import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.metrics import euclidean_distances



nodes_df = pd.read_csv('customsmall_nopool_ranks.csv')
nodes_wide_df = nodes_df.pivot(index = 'filter_num',columns='class', values='prune_score')


def get_layer(filter_num, df = nodes_df, col = 'layer'):
    return df.loc[(df['filter_num'] == filter_num) & (df['class'] == df['class'].unique()[0]), col].item()


nodes_wide_df.reset_index(inplace=True)

nodes_wide_df['layer'] = nodes_wide_df['filter_num'].apply(get_layer)

classes = nodes_df['class'].unique()


layers = {}
for index, row in nodes_df[nodes_df['class'] == 'overall'].iterrows():
    if row['layer'] not in layers:
        layers[row['layer']] = []
    layers[row['layer']].append(row['filter_num'])

labels_list=(list(nodes_df[nodes_df['class'] == 'overall'].filter_num))
layers_list=(list(nodes_df[nodes_df['class'] == 'overall'].layer))

layer_similarities = {}

for layer in layers:
	
	layer_similarities[layer] = euclidean_distances(nodes_wide_df[nodes_wide_df['layer'] == layer].iloc[:,1:-1])

print(layer_similarities[1].shape)


layer_mds = {}
for layer in layer_similarities:
	print('layer: %s'%str(layer))
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, 
      random_state=2, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(layer_similarities[layer]).embedding_
	layer_mds[layer] = pos

print(layer_mds)