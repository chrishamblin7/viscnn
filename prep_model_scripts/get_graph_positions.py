#generate mds projections of nodes layerwise, as determined by their per category rank scores
import os
import time
import torch
import pickle
import argparse
import sys

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("output_folder", type = str, help='the folder name for this prepped model')
	args = parser.parse_args()
	return args

args = get_args()
output_folder = args.output_folder


sys.path.insert(0, os.path.abspath('../'))
sys.path.insert(0, os.path.abspath('../visualizer_scripts/'))

os.chdir('../')
from visualizer_helper_functions import rank_file_2_df
os.chdir('./prep_model_scripts')

print('generating mds projection of nodes')

import numpy as np
import pandas as pd
from sklearn import manifold
from sklearn.metrics import euclidean_distances


categories_nodes_df = pd.read_csv('../prepped_models/%s/ranks/categories_nodes_ranks.csv'%output_folder)
overall_edge_df = rank_file_2_df('../prepped_models/%s/ranks/categories_edges/overall_edges_rank.pt'%output_folder)

misc_data = pickle.load(open('../prepped_models/%s/misc_graph_data.pkl'%output_folder,'rb'))
layer_nodes = misc_data['layer_nodes']
num_layers = misc_data['num_layers']
num_nodes = misc_data['num_nodes']
categories = misc_data['categories']
num_img_chan = misc_data['num_img_chan']
imgnode_positions = misc_data['imgnode_positions']
imgnode_colors = misc_data['imgnode_colors']
imgnode_names = misc_data['imgnode_names']


#make wide version of nodes_df
def get_col(node_num, df = categories_nodes_df, idx = 'node_num', col = 'layer'):
    return df.loc[(df[idx] == node_num) & (df['category'] == df['category'].unique()[0]), col].item()

def add_norm_col(df,categories=categories,delete_columns = ['small_SPAN','overall']):
    for c in delete_columns:
        try:
            categories.remove(c)
        except:
            pass
    norms = []
    for index, row in df.iterrows():
        norm = 0
        for i,category in enumerate(categories):
            norm += row[category]**2
        norm = np.sqrt(norm)
        norms.append(norm)
    norms = np.array(norms)
    df['category_norm'] = norms
    return df


def gen_wide_df(rank_type,df=categories_nodes_df,delete_columns = ['small_SPAN','overall']):
    print('making wide version of df')
    nodes_wide_df = df.pivot(index = 'node_num',columns='category', values=rank_type)
    nodes_wide_df.reset_index(inplace=True)
    nodes_wide_df['layer'] = nodes_wide_df['node_num'].apply(get_col)
    nodes_wide_df = nodes_wide_df.rename(columns = {'category':'index'})
    for c in delete_columns:
        try:
            nodes_wide_df = nodes_wide_df.drop([c], axis=1)
        except:
            print('could drop column %s'%c)
    nodes_wide_df = add_norm_col(nodes_wide_df,delete_columns = delete_columns) 
    return nodes_wide_df

#rotation for mds plots
from scipy.spatial.distance import cdist

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


def rotate_cartesian(vec2d,r):    #rotates 2d cartesian coordinates by some radians 
    x,y = vec2d[0], vec2d[1]
    x_out = np.sqrt(x**2+y**2)*np.cos(np.arctan2(y,x)+r)
    y_out = np.sqrt(x**2+y**2)*np.sin(np.arctan2(y,x)+r)
    return np.array([x_out,y_out])

def rotate_mds(layer_mds,rank_type,imgnode_positions=imgnode_positions,max_edges = 40,angles_tested=64):
    print('rotating layers to minimize edge lengths')
    for layer in range(len(layer_mds)):
        all_layer_positions = layer_mds[layer]
        layer_df = overall_edge_df.loc[(overall_edge_df['layer']==layer)].sort_values(rank_type+'_rank',ascending=False).head(max_edges)
        if layer == 0:
            all_prev_layer_positions = np.swapaxes(np.array([imgnode_positions['Y'],imgnode_positions['Z']]),0,1)
        else:
            all_prev_layer_positions = layer_mds[layer-1]
        #gen positions matrix for important edges
        select_layer_positions = []
        select_prev_layer_positions = []
        for row in layer_df.itertuples():
            select_layer_positions.append(all_layer_positions[row.out_channel])
            select_prev_layer_positions.append(all_prev_layer_positions[row.in_channel])
        #go through discrete rotations and find min distance
        min_dist = 10000000
        min_discrete_angle = 0
        for p in range(0,angles_tested):
            test_layer_positions=np.apply_along_axis(rotate_cartesian, 1, select_layer_positions,r=p*2*np.pi/angles_tested)
            dist = sum(np.diagonal(cdist(test_layer_positions,select_prev_layer_positions)))
            if dist < min_dist:
                min_discrete_angle = p
                min_dist = dist
        #update layer mds at layer by rotating by optimal angle
        print('rotating layer %s by %s rads'%(str(layer),str(min_discrete_angle*2*np.pi/angles_tested)))
        layer_mds[layer] = np.apply_along_axis(rotate_cartesian, 1, layer_mds[layer],r=min_discrete_angle*2*np.pi/angles_tested)
    return layer_mds 


def gen_layer_mds(nodes_df=categories_nodes_df,delete_columns = ['small_SPAN','overall']):
    mds_projections ={}
    for rank_type in ['actxgrad','act','grad']:
        nodes_wide_df = gen_wide_df(rank_type+'_rank',df=nodes_df,delete_columns = delete_columns)
        layer_similarities = {}
        for layer in range(len(layer_nodes)):
            layer_df = nodes_wide_df[nodes_wide_df['layer'] == layer]
            for category in categories:
                layer_df[category] = layer_df.apply(lambda row : row[category]/row['category_norm'], axis = 1)   
            layer_similarities[layer] = euclidean_distances(layer_df.iloc[:,1:-2])

        layer_mds = {}
        for layer in layer_similarities:
            print('layer: %s'%str(layer))
            mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, 
              random_state=2, dissimilarity="precomputed", n_jobs=1)
            pos = mds.fit(layer_similarities[layer]).embedding_
            layer_mds[layer] = pos
        layer_mds = rotate_mds(layer_mds,rank_type)
        mds_projections[rank_type] = layer_mds
    return mds_projections


#grid layer projection

def gen_grid_positions():
    grid_projections = {}
    for layer in range(len(layer_nodes)):
        grid_projections[layer] = []
        num_nodes = len(layer_nodes[layer][1])
        if num_nodes == 1:
            grid_projections[layer] = np.array([[0,0]])
            continue
        elif num_nodes == 2:
            grid_projections[layer] = np.array([[.1,0],
                                                [-.1,0]])
            continue
        elif num_nodes == 3:
            grid_projections[layer] = np.array([[.1,.1],
                                                [0,0],
                                                [-.1,-.1]])
            continue
        elif num_nodes == 4:
            grid_projections[layer] = np.array([[.1,.1],
                                                [-.1,.1],
                                                [.1,-.1],
                                                [-.1,-.1]])
            continue
        elif num_nodes == 5:
            grid_projections[layer] = np.array([[.1,.1],
                                                [-.1,.1],
                                                [0,0],
                                                [.1,-.1],
                                                [-.1,-.1]])
            continue
        elif num_nodes == 6:
            grid_projections[layer] = np.array([[.1,.1],
                                                [0,.1],
                                                [-.1,.1],
                                                [.1,-.1],
                                                [0,-.1],
                                                [-.1,-.1]])
            continue
        elif num_nodes == 7:
            grid_projections[layer] = np.array([[.1,.1],
                                                [0,.1],
                                                [-.1,.1],
                                                [0,0],
                                                [.1,-.1],
                                                [0,-.1],
                                                [-.1,-.1]])
            continue
        elif num_nodes == 8:
            grid_projections[layer] = np.array([[.1,.1],
                                                [0,.1],
                                                [-.1,.1],
                                                [-.1,0],
                                                [.1,0],
                                                [.1,-.1],
                                                [0,-.1],
                                                [-.1,-.1]])  
            continue
        elif num_nodes == 9:
            grid_projections[layer] = np.array([[.1,.1],
                                                [0,.1],
                                                [-.1,.1],
                                                [-.1,0],
                                                [0,0],
                                                [.1,0],
                                                [.1,-.1],
                                                [0,-.1],
                                                [-.1,-.1]]) 
            continue
            
        elif num_nodes < 20:
            max_dis = .2
        elif num_nodes < 40:
            max_dis = .3
        elif num_nodes < 60:
            max_dis = .4
        elif num_nodes < 80:
            max_dis = .5
        elif num_nodes < 100:
            max_dis = .6
        elif num_nodes < 120:
            max_dis = .7
        elif num_nodes < 140:
            max_dis = .8
        else:
            max_dis = 1
        if np.floor(np.sqrt(num_nodes))*np.ceil(np.sqrt(num_nodes)) < num_nodes:
            x_spaces, y_spaces = np.ceil(np.sqrt(num_nodes)),np.ceil(np.sqrt(num_nodes))
        else:
            x_spaces, y_spaces = np.floor(np.sqrt(num_nodes)),np.ceil(np.sqrt(num_nodes))
        x = np.linspace(max_dis,-1*max_dis,int(x_spaces))
        y = np.linspace(max_dis,-1*max_dis,int(y_spaces))
        X,Y = np.meshgrid(x,y)
        X_flat = [item for sublist in X for item in sublist]
        Y_flat = [item for sublist in Y for item in sublist]
        for i in range(num_nodes):
            grid_projections[layer].append([X_flat[i],Y_flat[i]])    
        grid_projections[layer] = np.array(grid_projections[layer])
    return grid_projections


##Force Field equations
#center of mass
def points_2_com(points):
    #in_arr.shape = (num_points,dimensionality of points)
    #outputs center of mass as np.array([dim1,dim2, . . . ])
    outlist = []
    for dim in range(points.shape[1]):
        outlist.append(np.mean(points[:,dim]))
    return np.array(outlist)

def distance_and_direction(p1,p2):
    #returns a scalar magnitude and direction vector [x,y] from p1 [x,y] to p2[x,y]
    distance = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    direction = [p1[0]-p2[0],p1[1]-p2[1]]
    return distance, direction

def magnitude(point):
    return np.sqrt(np.sum(point**2))

def euc_2_polar(points):
    x,y = points[:,0], points[:,1]
    r = np.sqrt(x**2+y**2)
    t = np.arctan2(y,x)
    return r,t

def polar_2_euc(r,t):
    x = r*np.cos(t)
    y = r*np.sin(t)
    points = np.array([x,y]).transpose()
    return points

def scale_points(points,maximum = .25):
    r,t = euc_2_polar(points)
    max_r = np.max(r)
    scale = maximum/max_r
    r_scaled = scale*r
    scaled_points = polar_2_euc(r_scaled,t)
    return scaled_points
    

def scaling_forces(points):
    #in_arr.shape = (num_points,dimensionality of points)
    #outputs scalar forcing scaling factor
    distances = euclidean_distances(points)
    m = np.min(distances[np.triu_indices(points.shape[0], k = 1)])
    return m/2
    

def points_2_nonlin_repel(points, scaling):
    #in_arr.shape = (num_points,dimensionality of points)
    #outputs array of same shape with (num_points, dimensions of force vectors on those point)
    distances = euclidean_distances(points)
    mean_dis = np.mean(distances)
    #max_dis = np.max(distances)
    def force_on_point(point, points = points,scale = 1/mean_dis):
        dif = points - point
        dif_r, dif_t = euc_2_polar(dif)
        same_point_indices = np.where(dif_r == 0)[0]
        dif_r = np.delete(dif_r, same_point_indices, axis=0)
        dif_t = np.delete(dif_t, same_point_indices, axis=0)
        #magnitude = np.apply_along_axis(magnitude, 1, dif) 
        #force_mag = scaling*(1-scale*dif_r)
        #import pdb; pdb.set_trace()
        force_mag = 1/(dif_r)*mean_dis
        force_vecs = -1*polar_2_euc(force_mag,dif_t)
        #force_vecs = -1*np.delete(force_vecs, same_point_indices, axis=0)
        return np.sum(force_vecs,axis=0) 
    return np.apply_along_axis(force_on_point, 1, points)
    
    
def points_2_com_nonlin_attract(points,scaling):
    #in_arr.shape = (num_points,dimensionality of points)
    #outputs array of same shape with (num_points, dimensions of force vectors on those point)
    com = points_2_com(points)
    num_points = points.shape[0]
    def com_force_on_point(point, com = com, num_points = num_points,scaling=scaling):
        return [(com[0]-point[0])*scaling*num_points,(com[1]-point[1])*scaling*num_points]
    return np.apply_along_axis(com_force_on_point, 1, points)
    
    

def points_2_nonlin_force_field(points, scaling=1):
    #in_arr.shape = (num_points,dimensionality of points)
    #outputs array of same shape with (num_points, dimensions of force vectors on those point)
    scaling = scaling*scaling_forces(points)
    #return points_2_nonlin_repel(points,scaling)
    return points_2_nonlin_repel(points,scaling)+points_2_com_nonlin_attract(points,scaling)

def nonlinear_push_points(points,rate=.1):
    forces = points_2_nonlin_force_field(points)
    return points + rate*forces

def gen_layer_smooth_mds(mds_projections, iterations = 100,init_rate = .0001,step_ratio = 10,scale = .25):
    print('smoothing mds plots')
    init_rate = .0001
    smooth_mds_projection = {}
    #import pdb; pdb.set_trace()
    for rank_type in mds_projections.keys():
        smooth_mds_projection[rank_type] = {}
        for j in mds_projections[rank_type]:
            points = mds_projections[rank_type][j]
            for i in range(1,iterations):
                #move_points 
                #prev_points = deepcopy(points)
                points = nonlinear_push_points(points,rate=np.min([init_rate+i*init_rate/step_ratio,init_rate*10]))
            smooth_mds_projection[rank_type][j] = points
    
    max_points = 0
    for rank_type in mds_projections.keys():
        for i in mds_projections[rank_type]:
            points = mds_projections[rank_type][i]
            if points.shape[0] > max_points:
                max_points = points.shape[0]
                max_layer = i
        #get layer with maximum points and scale
        for i in mds_projections[rank_type]:
            if i == max_layer:
                smooth_mds_projection[rank_type][i] = scale_points(smooth_mds_projection[rank_type][i],maximum = scale)
            else:
                smooth_mds_projection[rank_type][i] = scale_points(smooth_mds_projection[rank_type][i],maximum = scale*np.sqrt(float(smooth_mds_projection[rank_type][i].shape[0])/float(max_points)))
        #rotate plots
        smooth_layer_mds = rotate_mds(smooth_mds_projection[rank_type],rank_type)
        smooth_mds_projection[rank_type] = smooth_layer_mds
    return smooth_mds_projection


        

    


grid_projections = gen_grid_positions()       
mds_projections = gen_layer_mds()
mds_smooth_projections = gen_layer_smooth_mds(mds_projections)



all_node_positions_unformatted = {'MDS':mds_projections,'Grid':grid_projections,'MDS smooth':mds_smooth_projections}

def format_node_positions(projection='MDS',rank_type = 'actxgrad'):
    layer_distance = 1   # distance in X direction each layer is separated by
    node_positions = []
    layer_offset = 0
    if projection == 'MDS':
        unformatted = all_node_positions_unformatted['MDS'][rank_type]
    elif projection == 'MDS smooth':
        unformatted = all_node_positions_unformatted['MDS smooth'][rank_type]
    else:
        unformatted = all_node_positions_unformatted['Grid']
    for layer in unformatted:
        node_positions.append({})
        node_positions[-1]['X'] = [] 
        node_positions[-1]['Y'] = [] 
        node_positions[-1]['Z'] = []  
        for i in range(len(unformatted[layer])): 
            node_positions[-1]['Y'].append(unformatted[layer][i][0])
            node_positions[-1]['Z'].append(unformatted[layer][i][1])
            node_positions[-1]['X'].append(layer_offset)
        layer_offset+=1*layer_distance
    return node_positions



all_node_positions_formatted = {'MDS':{},'MDS smooth':{}}

for rank_type in ['actxgrad','act','grad']:
    all_node_positions_formatted['MDS'][rank_type] =  format_node_positions(projection = 'MDS',rank_type = rank_type) 
for rank_type in ['actxgrad','act','grad']:
    all_node_positions_formatted['MDS smooth'][rank_type] =  format_node_positions(projection = 'MDS smooth',rank_type = rank_type) 

all_node_positions_formatted['Grid'] = format_node_positions(projection = 'Grid') 

pickle.dump(all_node_positions_formatted, open('../prepped_models/%s/node_positions.pkl'%output_folder,'wb'))